import { randomUUID } from 'crypto';
import fs from 'fs';
import toml from 'toml';
import dotenv from 'dotenv';
import paths from './paths.js';

// Centralized dotenv bootstrap: load root .env via paths helper
dotenv.config({ path: paths.resolveRoot('.env') });

export interface AppConfig {
	discordToken: string;
	voiceChannelId?: string;
	aiServiceUrl: string; // ws://host:port
	sessionName: string;
	chunkMs: number;
	allowedCommanders: string[];
}

function loadTomlConfig(): any | undefined {
	const filePath = paths.resolveRoot('config.toml');
	try {
		const raw = fs.readFileSync(filePath, 'utf8');
		return toml.parse(raw);
	} catch (err) {
		console.warn(`[CONFIG] Warning: Could not parse config.toml: ${err}`);
		return undefined;
	}
}

function extractFromToml(t: any | undefined): Partial<AppConfig> {
	if (!t) return {};
	const out: Partial<AppConfig> = {};
	const discord = t.discord || {};
	const net = t.net || {};
	out.voiceChannelId = discord.voice_channel_id;
	out.aiServiceUrl = net.ai_service_url;
	out.chunkMs = net.chunk_ms;
	out.allowedCommanders = discord.allowed_commanders;
	return out;
}

export function loadConfig(parsed: { [key: string]: unknown }): AppConfig {
	const fileCfg = extractFromToml(loadTomlConfig());
	// Secrets only from environment variables.
	const discordToken = process.env.DISCORD_TOKEN || '';
	if (!discordToken) {
		throw new Error('DISCORD_TOKEN env var is required (do not store tokens in config files)');
	}

	// Precedence: CLI args > config.toml > defaults
	// NOTE: voice channel must be provided via CLI (required) to avoid ambiguous config file usage.
	const voiceChannelId = (parsed.voice as string | undefined) || undefined;
	if (!voiceChannelId) {
		throw new Error('Voice channel ID is required.');
	}
	const aiServiceUrl =
		(parsed['aiServiceUrl'] as string | undefined) ||
		fileCfg.aiServiceUrl ||
		'ws://localhost:8771';
	const sessionName =
		(parsed['sessionName'] as string | undefined) ||
		fileCfg.sessionName ||
		new Date().toISOString().replace(/[:.]/g, '-');
	const chunkMs = (parsed['chunk-ms'] as number | undefined) || fileCfg.chunkMs || 1000;
	const allowedCommanders =
		(parsed.allowedCommanders as string[] | undefined) || fileCfg.allowedCommanders || [];

	return { discordToken, voiceChannelId, aiServiceUrl, sessionName, chunkMs, allowedCommanders };
}

export function newSessionId(): string {
	return randomUUID();
}
