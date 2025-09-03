import {
	AudioSegmentMessage,
	WrapupResponseMessage,
	TranscriptionMessage,
	WrapupRequestMessage,
	ErrorMessage,
} from './messages.js';
import { PerUserVoiceSegmenter, VoiceSegment } from './voiceReceiver.js';
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
	sessionName: string;
	aiServiceUrl: string;
	userDirectory: UserDirectory;
	logsPath: string;
	wrapupsPath: string;
	profile?: string;
	// Overrides resolved from config/profile, passed to Python directly
	userIdMap?: Record<string, string>;
	phraseMap?: Record<string, string>;
	wrapupPrompt?: string;
	wrapupTips?: string[];
	asrPrompt?: string;
	// Segmenter options
	vadDbThreshold?: number;
	silenceGapMs?: number;
	vadFrameMs?: number;
	maxSegmentMs?: number;
}

export class Session {
	private readonly sessionInfo: SessionConfig;
	private readonly segmenter: PerUserVoiceSegmenter;
	private readonly websocketClient: PythonWsClient;
	private readonly logStream?: fs.WriteStream;
	private readonly logFilePath: string;
	private pendingWrapups: Map<string, Message> = new Map();

	constructor(sessionInfo: SessionConfig) {
		this.sessionInfo = sessionInfo;
		debug('New session created:', this.sessionInfo);

		this.segmenter = new PerUserVoiceSegmenter({
			send: this.handleAudioSegment,
			vadDbThreshold: this.sessionInfo.vadDbThreshold,
			silenceGapMs: this.sessionInfo.silenceGapMs,
			vadFrameMs: this.sessionInfo.vadFrameMs,
			maxSegmentMs: this.sessionInfo.maxSegmentMs,
		});

		this.websocketClient = new PythonWsClient(sessionInfo.aiServiceUrl, (msg) => {
			if (msg.type === 'transcription') {
				this.handleTranscription(msg as TranscriptionMessage);
			} else if (msg.type === 'error') {
				this.handleErrorMessage(msg as ErrorMessage);
			} else if (msg.type === 'wrapup.response') {
				this.handleWrapupResponse(msg as WrapupResponseMessage);
			}
		});

		// Ensure logs directory exists (project root /logs)
		const logsDir = this.sessionInfo.logsPath;
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

	public getSegmenter() {
		return this.segmenter;
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

	private handleAudioSegment = (segment: VoiceSegment) => {
		// Convert internal VoiceSegment -> AudioSegmentMessage for IPC
		// Warm user directory mapping in the background; don't block audio send.
		void this.sessionInfo.userDirectory.ensureDisplayName(
			segment.userId,
			this.sessionInfo.guildId,
		);
		const audioMsg: AudioSegmentMessage = {
			v: 1,
			type: 'audio.segment',
			id: segment.userId,
			index: segment.index,
			pcm_format: {
				sr: segment.pcmFormat.sampleRate,
				channels: segment.pcmFormat.channels,
				sample_width: segment.pcmFormat.sampleWidth,
			},
			started_ts: segment.startedTs,
			capture_ts: segment.captureTs,
			data_b64: pcm16ToBase64(segment.pcm16),
			// Per-job ASR prompt override computed on the JS side
			prompt: this.sessionInfo.asrPrompt,
		};
		this.websocketClient.send(audioMsg);
		debug(`Sent audio segment ${segment.index} for user ${segment.userId}`);
	};

	private handleTranscription = (segment: TranscriptionMessage) => {
		// Prefer directory display name; fall back to id when unknownreceived
		const display = this.sessionInfo.userDirectory.getDisplayNameSync(
			segment.id,
			this.sessionInfo.guildId,
		);
		const userName = display && display.length ? display : segment.id;
		debug('Received transcription:', segment);
		// Append JSONL line to log file
		if (this.logStream) {
			const out: JsonlLogEntry = {
				userId: segment.id,
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

	private handleErrorMessage = async (err: ErrorMessage) => {
		debug('Received error message from Python service:', err);
	};

	public handleWrapup = async (message: Message) => {
		const requestId = message.id;
		this.pendingWrapups.set(requestId, message);
		debug(`Wrapup command handled. Request ID: ${requestId}`);

		const entries = await this.readLogEntries();
		// Use the timestamp of the first log entry if available; otherwise use current time
		const startTs = entries && entries.length > 0 ? entries[0].startTs : Date.now() / 1000;
		// Build wire entries, then apply configured username/phrase maps on the JS side
		const transformed = entries.map((e) =>
			toWrapupLogEntry(e, this.sessionInfo.userIdMap, this.sessionInfo.phraseMap),
		);

		const wrapupReq: WrapupRequestMessage = {
			v: 1,
			type: 'wrapup.request',
			session_name: this.sessionInfo.sessionName,
			start_ts: startTs,
			log_entries: transformed,
			request_id: requestId,
			// userid_map/phrase_map applied in JS to avoid double application downstream
			wrapup_prompt: this.sessionInfo.wrapupPrompt,
			wrapup_tips: this.sessionInfo.wrapupTips,
		};
		this.websocketClient.send(wrapupReq);
		debug('Sent wrapup request to Python service.');
	};

	private handleWrapupResponse = async (response: WrapupResponseMessage) => {
		// Try to correlate using request_id if present
		let message: Message | undefined;
		debug('Received wrapup response:', response);
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

		// Write the raw outline unmodified to a markdown file and reply with the file attached.
		try {
			const wrapupsDir = this.sessionInfo.wrapupsPath;
			paths.ensureDir(wrapupsDir);
			const outPath = path.join(wrapupsDir, `${this.sessionInfo.sessionName}.md`);
			// Write the outline exactly as received (no decoration)
			fs.writeFileSync(outPath, response.outline, 'utf8');

			const attachment = new AttachmentBuilder(Buffer.from(response.outline, 'utf-8'), {
				name: `${this.sessionInfo.sessionName}.md`,
			});

			const replyContent = `Session wrap-up attached: ${this.sessionInfo.sessionName}`;
			await message.reply({ content: replyContent, files: [attachment] });
		} catch (e) {
			console.error('Failed to send wrapup reply to Discord or write file:', e);
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
