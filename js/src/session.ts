import {
	AudioChunkMessage,
	SummarizeResponseMessage,
	TranscriptionMessage,
	WrapupRequestMessage,
} from './messages.js';
import { PerUserVoiceChunker, VoiceChunk } from './voiceReceiver.js';
import { UserDirectory } from './userDirectory.js';
import { pcm16ToBase64 } from './audioUtils.js';
import fs from 'fs';
import path from 'path';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines, JsonlLogEntry } from './logs.js';
import { toWrapupLogEntry } from './messages.js';
import { PythonWsClient } from './websocketClient.js';
import { Message, AttachmentBuilder } from 'discord.js';
import { debug } from './debug.js';

export interface SessionConfig {
	sessionId: string;
	guildId?: string;
	voiceChannelId: string;
	startTimestamp: number; // epoch seconds (float)
	sessionName: string;
	aiServiceUrl: string;
	userDirectory: UserDirectory;
}

export class Session {
	private readonly sessionInfo: SessionConfig;
	private readonly chunker: PerUserVoiceChunker;
	private readonly websocketClient: PythonWsClient;
	private readonly logStream?: fs.WriteStream;
	private readonly logFilePath: string;
	private pendingWrapups: Map<string, Message> = new Map();

	constructor(sessionInfo: SessionConfig) {
		this.sessionInfo = sessionInfo;
		debug('New session created:', this.sessionInfo);

		this.chunker = new PerUserVoiceChunker(
			{
				targetSampleRate: 16000,
				chunkMs: 1000,
			},
			this.handleAudioChunk,
			sessionInfo.userDirectory,
			sessionInfo.guildId,
		);

		this.websocketClient = new PythonWsClient(sessionInfo.aiServiceUrl, (msg) => {
			if (msg.type === 'transcription') {
				this.handleTranscription(msg as TranscriptionMessage);
			} else if (msg.type === 'summarize.response') {
				this.handleSummarizeResponse(msg as SummarizeResponseMessage);
			}
		});

		// Ensure logs directory exists (project root /logs)
		const logsDir = paths.resolveRoot('logs');
		paths.ensureDir(logsDir);
		this.logFilePath = path.join(logsDir, `${this.sessionInfo.sessionId}.jsonl`);
		this.logStream = fs.createWriteStream(this.logFilePath, { flags: 'a', encoding: 'utf8' });
		// Fail fast on any file stream error — we own consistency for session logs
		this.logStream.on('error', (err) => {
			console.error('Fatal: session log stream error. Exiting.', err);
			debug('Log stream error:', err);
			process.exit(1);
		});
	}

	public getSessionId() {
		return this.sessionInfo.sessionId;
	}

	public getVoiceChannelId() {
		return this.sessionInfo.voiceChannelId;
	}

	public start() {
		this.websocketClient.start();
		debug('Session started, websocket client connected.');
	}

	public stop() {
		this.websocketClient.stop();
		debug('Session stopped, websocket client disconnected.');
		try {
			this.logStream?.end();
		} catch (e) {
			console.error('Error closing session log stream:', e);
		}
	}

	public getChunker() {
		return this.chunker;
	}

	public async logTextMessage(message: Message) {
		const displayName =
			(await this.sessionInfo.userDirectory.ensureDisplayName(
				message.author.id,
				this.sessionInfo.guildId,
			)) ?? message.author.displayName;

		debug(`Logging text message from ${displayName}: ${message.content}`);

		if (this.logStream) {
			const ts = message.createdTimestamp / 1000;
			const out: JsonlLogEntry = {
				userId: message.author.id,
				displayName,
				startTs: ts,
				endTs: ts,
				origin: 'text',
				text: message.content,
			};
			this.logStream.write(JSON.stringify(out) + '\n');
		}
	}

	private handleAudioChunk = (chunk: VoiceChunk) => {
		// Convert internal VoiceChunk -> AudioChunkMessage for IPC
		const audioMsg: AudioChunkMessage = {
			v: 1,
			type: 'audio.chunk',
			user_id: chunk.userId,
			index: chunk.index,
			pcm_format: {
				sr: chunk.pcmFormat.sampleRate,
				channels: chunk.pcmFormat.channels,
				sample_width: chunk.pcmFormat.sampleWidth,
			},
			started_ts: chunk.startedTs,
			capture_ts: chunk.captureTs,
			data_b64: pcm16ToBase64(chunk.pcm16),
		};
		this.websocketClient.send(audioMsg);
		debug(`Sent audio chunk ${chunk.index} for user ${chunk.userId}`);
	};

	private handleTranscription = (segment: TranscriptionMessage) => {
		// Prefer directory display name; fall back to user_id when unknown
		const display = this.sessionInfo.userDirectory.getDisplayNameSync(
			segment.user_id,
			this.sessionInfo.guildId,
		);
		const userName = display && display.length ? display : segment.user_id;
		debug('Received transcription:', segment);
		// Append JSONL line to log file
		if (this.logStream) {
			const out: JsonlLogEntry = {
				userId: segment.user_id,
				displayName: userName,
				startTs: segment.capture_ts,
				endTs: segment.end_ts,
				origin: 'voice',
				text: segment.text,
			};
			// any async error will trigger the stream 'error' handler and exit
			this.logStream.write(JSON.stringify(out) + '\n');
		}
	};

	public handleWrapup = async (message: Message) => {
		const requestId = message.id;
		this.pendingWrapups.set(requestId, message);
		debug(`Wrapup command handled. Request ID: ${requestId}`);

		const entries = await this.readLogEntries();
		const wrapupReq: WrapupRequestMessage = {
			v: 1,
			type: 'summarize.request',
			session_name: this.sessionInfo.sessionName,
			start_ts: this.sessionInfo.startTimestamp,
			log_entries: entries.map(toWrapupLogEntry),
			request_id: requestId,
		};
		this.websocketClient.send(wrapupReq);
		debug('Sent wrapup request to Python service.');
	};

	private handleSummarizeResponse = async (response: SummarizeResponseMessage) => {
		// Try to correlate using request_id if present
		let message: Message | undefined;
		debug('Received summarize response:', response);
		if ((response as any).request_id) {
			message = this.pendingWrapups.get((response as any).request_id);
			if (message) {
				this.pendingWrapups.delete((response as any).request_id);
				debug(`Found pending wrapup for request ID: ${(response as any).request_id}`);
			}
		}
		if (!message) message = this.pendingWrapups.values().next().value;
		if (!message) {
			console.warn('Received a summary response with no pending wrapup command.');
			debug('No pending wrapup command found for summary response.');
			return;
		}

		const replyContent = `
**Session Wrap-up for: ${this.sessionInfo.sessionName}**

**Transcript:**
\`\`\`
${response.transcript}
\`\`\`

**Outline:**
${response.outline}
`;

		try {
			await message.reply(replyContent);
		} catch (e) {
			console.error('Failed to send wrapup reply to Discord:', e);
		}
	};

	public handleLogCommand = async (message: Message) => {
		debug('Log command handled.');
		const entries = await this.readLogEntries();
		const logContent = formatLogLines(entries as any);

		const attachment = new AttachmentBuilder(
			Buffer.from(logContent || 'Log is empty.', 'utf-8'),
			{
				name: `${this.sessionInfo.sessionId}_log.txt`,
			},
		);

		try {
			await message.reply({
				content: `**Session Log for: ${this.sessionInfo.sessionName}**`,
				files: [attachment],
			});
		} catch (e) {
			console.error('Failed to send log reply to Discord:', e);
		}
	};

	private async readLogEntries(): Promise<JsonlLogEntry[]> {
		debug(`Reading log entries from: ${this.logFilePath}`);
		return await readLogEntriesStrict(this.logFilePath);
	}
}
