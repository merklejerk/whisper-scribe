import {
	AudioChunkMessage,
	SummarizeResponseMessage,
	TranscriptionMessage,
	WrapupRequestMessage,
} from './messages.js';
import { PerUserVoiceChunker, VoiceChunk } from './voiceReceiver.js';
import { pcm16ToBase64 } from './audioUtils.js';
import fs from 'fs';
import path from 'path';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines } from './logs.js';
import { PythonWsClient } from './websocketClient.js';
import { Message } from 'discord.js';
import { debug } from './debug.js';

export interface SessionConfig {
	sessionId: string;
	guildId?: string;
	voiceChannelId: string;
	startTimestamp: number; // epoch seconds (float)
	sessionName: string;
	aiServiceUrl: string;
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
				chunkMs: 500,
			},
			this.handleAudioChunk,
		);

		this.websocketClient = new PythonWsClient(sessionInfo.aiServiceUrl, (msg) => {
			debug('Received message from Python service:', msg);
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

	private handleAudioChunk = (chunk: VoiceChunk) => {
		// Convert internal VoiceChunk -> AudioChunkMessage for IPC
		const audioMsg: AudioChunkMessage = {
			v: 1,
			type: 'audio.chunk',
			user_id: chunk.user_id,
			index: chunk.index,
			pcm_format: chunk.pcm_format,
			started_ts: chunk.started_ts,
			capture_ts: chunk.capture_ts,
			data_b64: pcm16ToBase64(chunk.pcm16),
		};
		this.websocketClient.send(audioMsg);
		debug(`Sent audio chunk ${chunk.index} for user ${chunk.user_id}`);
	};

	private handleTranscription = (segment: TranscriptionMessage) => {
		// TODO: Post to Discord channel
		const userName = 'TODO'; // TODO: look up from session
		console.log(`[${userName}]: ${segment.text}`);
		debug('Received transcription:', segment);
		// Append JSONL line to log file
		if (this.logStream) {
			const out = {
				user_id: segment.user_id,
				user_name: userName,
				// TODO: we need to reconstruct these from the audio chunks
				// start_ts: segment.start_ts,
				// end_ts: segment.end_ts,
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
			log_entries: entries as any[],
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

		const replyContent = `
**Session Log for: ${this.sessionInfo.sessionName}**
\`\`\`
${logContent || 'Log is empty.'}
\`\`\`
`;

		try {
			await message.reply(replyContent);
		} catch (e) {
			console.error('Failed to send log reply to Discord:', e);
		}
	};

	private async readLogEntries(): Promise<any[]> {
		debug(`Reading log entries from: ${this.logFilePath}`);
		return await readLogEntriesStrict(this.logFilePath);
	}
}
