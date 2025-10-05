import { AudioSegmentMessage, TranscriptionMessage, ErrorMessage } from './messages.js';
import type { VoiceSegment } from './voiceReceiver.js';
import { UserDirectory } from './userDirectory.js';
import { pcm16ToBase64 } from './audioUtils.js';
import fs from 'fs';
import path from 'path';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines, JsonlLogEntry } from './logs.js';
import { PythonWsClient } from './websocketClient.js';
import { Message, AttachmentBuilder } from 'discord.js';
import { debug } from './debug.js';
import { createWrapup } from './wrapup.js';
import { createSessionGist } from './gist.js';

export interface SessionConfig {
	sessionId: string;
	guildId?: string;
	voiceChannelId: string;
	sessionName: string;
	aiServiceUrl: string;
	userDirectory: UserDirectory;
	profile?: string;
	userIdMap?: Record<string, string>;
	phraseMap?: Record<string, string>;
	wrapupPrompt?: string;
	wrapupTips?: string[];
	wrapupVocabulary?: string[];
	wrapupModel?: string;
	wrapupTemperature?: number;
	wrapupMaxTokens?: number;
	geminiApiKey?: string;
	githubToken?: string;
	asrPrompt?: string;
	// Rolling ASR context configuration (optional)
	asrContextWords?: number; // default 40
	prevWrapupRef?: string; // optional session name or URL for prior wrapup context
	gist?: boolean; // if true, upload gist on !wrapup
}

export class Session {
	private readonly sessionInfo: SessionConfig;
	private readonly websocketClient: PythonWsClient;
	private readonly logStream?: fs.WriteStream;
	private readonly logFilePath: string;
	// Rolling context buffer across text + voice
	private contextWords: string[] = [];
	private readonly maxContextWords: number;

	constructor(sessionInfo: SessionConfig) {
		this.sessionInfo = sessionInfo;
		debug('New session created:', this.sessionInfo);

		// Configure limit for rolling context (by word count only)
		this.maxContextWords = Math.max(0, sessionInfo.asrContextWords ?? 40);

		this.websocketClient = new PythonWsClient(sessionInfo.aiServiceUrl, (msg) => {
			if (msg.type === 'transcription') {
				this.handleTranscription(msg as TranscriptionMessage);
			} else if (msg.type === 'error') {
				this.handleErrorMessage(msg as ErrorMessage);
			}
		});

		const sessionDir = paths.sessionDataDir(this.sessionInfo.sessionName);
		paths.ensureDir(sessionDir);
		this.logFilePath = path.join(sessionDir, 'log.jsonl');
		this.logStream = fs.createWriteStream(this.logFilePath, { flags: 'a', encoding: 'utf8' });
		// Fail fast on any file stream error — we own consistency for session logs
		this.logStream.on('error', (err) => {
			console.error('Fatal: session log stream error. Exiting.', err);
			debug('Log stream error:', err);
			process.exit(1);
		});
	}

	public getSessionName() {
		return this.sessionInfo.sessionName;
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

		// Update rolling context with user text
		this.appendContext(message.content);
	}

	public handleAudioSegment = (segment: VoiceSegment) => {
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
			// Per-job ASR prompt override computed on the JS side, with rolling context appended
			prompt: this.composePromptWithContext(),
		};
		this.websocketClient.send(audioMsg);
		debug(`Sent audio segment ${segment.index} for user ${segment.userId}`);
	};

	private handleTranscription = (segment: TranscriptionMessage) => {
		// We use the user ID as the segment ID in the `audio.segment` message.
		const userId = segment.id;
		// Prefer directory display name; fall back to id when unknownreceived
		const display = this.sessionInfo.userDirectory.getDisplayNameSync(
			userId,
			this.sessionInfo.guildId,
		);
		const displayName = display && display.length ? display : userId;
		debug(
			`Received ${(segment.end_ts - segment.capture_ts).toFixed(1)}s transcription from ${displayName}\n > "${segment.text}"\n  `,
		);
		// Append JSONL line to log file
		if (this.logStream) {
			const out: JsonlLogEntry = {
				userId,
				displayName,
				startTs: segment.capture_ts,
				endTs: segment.end_ts,
				origin: 'voice',
				text: segment.text,
			};
			// any async error will trigger the stream 'error' handler and exit
			this.logStream.write(JSON.stringify(out) + '\n');
		}

		// Update rolling context with recognized speech
		if (segment.text) this.appendContext(segment.text);
	};

	private handleErrorMessage = async (err: ErrorMessage) => {
		debug('Received error message from Python service:', err);
	};

	public handleWrapupCommand = async (message: Message) => {
		debug(`Wrapup command handled.`);
		const apiKey = this.sessionInfo.geminiApiKey || process.env.GEMINI_API_KEY || '';
		if (!apiKey) {
			try {
				await message.reply('Wrapup requires GEMINI_API_KEY to be set.');
			} catch {}
			return;
		}

		try {
			const outlineContent = await createWrapup({
				sessionName: this.sessionInfo.sessionName,
				apiKey,
				userIdMap: this.sessionInfo.userIdMap,
				phraseMap: this.sessionInfo.phraseMap,
				vocabulary: this.sessionInfo.wrapupVocabulary,
				model: this.sessionInfo.wrapupModel,
				prompt: this.sessionInfo.wrapupPrompt,
				temperature: this.sessionInfo.wrapupTemperature,
				maxOutputTokens: this.sessionInfo.wrapupMaxTokens,
				tips: this.sessionInfo.wrapupTips,
				prevWrapupRef: this.sessionInfo.prevWrapupRef,
			});

			if (!outlineContent) {
				await message.reply('No session log yet. Say something first.');
				return;
			}

			// Ensure we send the cached file path if available
			const wrapupPath = paths.sessionWrapupPath(this.sessionInfo.sessionName);
			let attachment: AttachmentBuilder;
			if (fs.existsSync(wrapupPath)) {
				attachment = new AttachmentBuilder(wrapupPath);
			} else {
				attachment = new AttachmentBuilder(Buffer.from(outlineContent, 'utf-8'), {
					name: `${this.sessionInfo.sessionName}.md`,
				});
			}
			const replyContent = `Session wrap-up attached: ${this.sessionInfo.sessionName}`;
			await message.reply({ content: replyContent, files: [attachment] });

			// Optionally upload to a private gist
			if (this.sessionInfo.gist) {
				try {
					const url = await createSessionGist(this.sessionInfo.sessionName, {
						token: this.sessionInfo.githubToken || process.env.GITHUB_TOKEN,
					});
					await message.reply(`Gist created: ${url}`);
				} catch (e: any) {
					console.error('Gist upload failed:', e);
					try {
						await message.reply(`Gist upload failed: ${e?.message || e}`);
					} catch {}
				}
			}
		} catch (err: any) {
			console.error('Wrapup generation failed:', err);
			const reply =
				err.message === 'No log entries found for session.'
					? 'No session log yet. Say something first.'
					: `Wrapup failed: ${err?.message || 'unknown error'}`;
			try {
				await message.reply(reply);
			} catch {}
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

	// ---- Context helpers ----
	private composePromptWithContext(): string | undefined {
		const base = this.sessionInfo.asrPrompt?.trim();
		const ctx = this.getContextText();
		return [base, ctx].filter(Boolean).join(' ');
	}

	private appendContext(text: string) {
		if (!text) return;
		const tokens = text
			.split(/\s+/)
			.map((t) => t.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ''))
			.filter((t) => t.length > 0);
		if (tokens.length === 0) return;
		this.contextWords.push(...tokens);
		if (this.contextWords.length > this.maxContextWords) {
			this.contextWords.splice(0, this.contextWords.length - this.maxContextWords);
		}
	}

	private getContextText(): string {
		if (this.contextWords.length === 0) return '';
		return this.contextWords.slice(-this.maxContextWords).join(' ');
	}
}
