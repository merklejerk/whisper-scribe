import { randomUUID } from 'crypto';
import fs from 'fs';
import toml from 'toml';
import dotenv from 'dotenv';
import paths from './paths.js';

// Centralized dotenv bootstrap: load root .env via paths helper
dotenv.config({ path: paths.resolveRoot('.env') });

export interface AppConfig {
	discordToken: string;
	aiServiceUrl: string; // ws://host:port
	allowedCommanders: string[];
	dataRoot: string; // new unified data root
	profile?: string;
	userIdMap?: Record<string, string>;
	phraseMap?: Record<string, string>;
	// Optional overrides derived from profile/base config
	wrapupPrompt?: string;
	wrapupTips?: string[];
	wrapupModel?: string;
	wrapupTemperature?: number;
	wrapupMaxTokens?: number;
	geminiApiKey?: string;
	asrPrompt?: string;
	// Segmenter (VAD) options
	vadDbThreshold?: number;
	silenceGapMs?: number;
	vadFrameMs?: number;
	maxSegmentMs?: number;
	minSegmentMs?: number;
}

export interface CliArgs {
	aiServiceUrl?: string;
	allowedCommanders?: string[];
	profile?: string;
}

export function loadTomlConfig(): any | undefined {
	const filePath = paths.resolveRoot('config.toml');
	try {
		const raw = fs.readFileSync(filePath, 'utf8');
		return toml.parse(raw);
	} catch (err) {
		console.warn(`[CONFIG] Warning: Could not parse config.toml: ${err}`);
		return undefined;
	}
}

function extractFromToml(t: any | undefined, profile?: string): Partial<AppConfig> {
	if (!t) return {};
	const out: Partial<AppConfig> = {};
	const discord = t.discord || {};
	const net = t.net || {};
	const voice = t.voice || {};
	const wrapup = t.wrapup || {};
	out.aiServiceUrl = net.ai_service_url;
	out.allowedCommanders = discord.allowed_commanders;
	// Voice/segmenter base options (not profile-overridable)
	if (typeof voice.vad_db_threshold === 'number') out.vadDbThreshold = voice.vad_db_threshold;
	if (typeof voice.silence_gap_ms === 'number') out.silenceGapMs = voice.silence_gap_ms;
	if (typeof voice.vad_frame_ms === 'number') out.vadFrameMs = voice.vad_frame_ms;
	if (typeof voice.max_segment_ms === 'number') out.maxSegmentMs = voice.max_segment_ms;
	if (typeof voice.min_segment_ms === 'number') out.minSegmentMs = voice.min_segment_ms;
	// Base-level defaults
	if (t.wrapup && typeof t.wrapup.prompt === 'string') out.wrapupPrompt = t.wrapup.prompt;
	if (t.wrapup && Array.isArray(t.wrapup.tips)) out.wrapupTips = t.wrapup.tips;
	if (typeof wrapup.model === 'string') out.wrapupModel = wrapup.model;
	if (typeof wrapup.temperature === 'number') out.wrapupTemperature = wrapup.temperature;
	if (typeof wrapup.max_output_tokens === 'number')
		out.wrapupMaxTokens = wrapup.max_output_tokens;
	if (t.whisper && typeof t.whisper.prompt === 'string') out.asrPrompt = t.whisper.prompt;

	if (profile) {
		const profiles = (t.profile || {}) as Record<string, any>;
		const p = profiles[profile];
		if (p && typeof p === 'object') {
			out.profile = profile;
			// Merge allowed_commanders.
			if (Array.isArray(p.allowed_commanders)) {
				out.allowedCommanders = [...(out.allowedCommanders ?? []), ...p.allowed_commanders];
			}
			// Merge tips.
			if (Array.isArray(p.wrapup_tips)) {
				out.wrapupTips = [...(out.wrapupTips ?? []), ...p.wrapup_tips];
			}
			// Merge userid_map.
			out.userIdMap = { ...(t.userid_map || {}), ...(p.userid_map || {}) };
			// Merge phrase_map.
			out.phraseMap = { ...(t.phrase_map || {}), ...(p.phrase_map || {}) };
			// Override wrapup prompt.
			if (typeof p.wrapup_prompt === 'string') out.wrapupPrompt = p.wrapup_prompt;
			// Override whisper prompt.
			if (typeof p.whisper_prompt === 'string') out.asrPrompt = p.whisper_prompt;
		}
	}
	return out;
}

export function loadConfig(parsed: CliArgs): AppConfig {
	const fileCfg = extractFromToml(loadTomlConfig(), parsed.profile);
	// Secrets only from environment variables.
	const discordToken = process.env.DISCORD_TOKEN || '';
	if (!discordToken) {
		throw new Error('DISCORD_TOKEN env var is required (do not store tokens in config files)');
	}

	// LLM API keys from environment (Node side)
	const geminiApiKey =
		process.env.GEMINI_API_KEY || process.env.gemini_api_key || (fileCfg as any).geminiApiKey || undefined;

	// Precedence: CLI args > config.toml > defaults
	const aiServiceUrl =
		(parsed.aiServiceUrl as string | undefined) ||
		fileCfg.aiServiceUrl ||
		'ws://localhost:8771';
	const allowedCommanders =
		(parsed.allowedCommanders as string[] | undefined) || fileCfg.allowedCommanders || [];

	// Compute absolute paths
	const dataRoot = paths.dataRoot();

	return {
		discordToken,
		aiServiceUrl,
		allowedCommanders,
		dataRoot,
		profile: fileCfg.profile,
		userIdMap: fileCfg.userIdMap,
		phraseMap: fileCfg.phraseMap,
		wrapupPrompt: fileCfg.wrapupPrompt,
		wrapupTips: fileCfg.wrapupTips,
		wrapupModel: fileCfg.wrapupModel,
		wrapupTemperature: fileCfg.wrapupTemperature,
		wrapupMaxTokens: fileCfg.wrapupMaxTokens,
		geminiApiKey,
		asrPrompt: fileCfg.asrPrompt,
		vadDbThreshold: fileCfg.vadDbThreshold,
		silenceGapMs: fileCfg.silenceGapMs,
		vadFrameMs: fileCfg.vadFrameMs,
		maxSegmentMs: fileCfg.maxSegmentMs,
		minSegmentMs: fileCfg.minSegmentMs,
	};
}

export function newSessionId(): string {
	return randomUUID();
}
