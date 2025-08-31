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
	chunkMs: number;
	allowedCommanders: string[];
	logsPath: string;
	wrapupsPath: string;
}

export interface ParsedArgs {
	aiServiceUrl?: string;
	allowedCommanders?: string[];
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

function extractFromToml(t: any | undefined): Partial<AppConfig> {
	if (!t) return {};
	const out: Partial<AppConfig> = {};
	const discord = t.discord || {};
	const net = t.net || {};
	out.aiServiceUrl = net.ai_service_url;
	out.chunkMs = net.chunk_ms;
	out.allowedCommanders = discord.allowed_commanders;
	return out;
}

export function loadConfig(parsed: ParsedArgs): AppConfig {
	const fileCfg = extractFromToml(loadTomlConfig());
	// Secrets only from environment variables.
	const discordToken = process.env.DISCORD_TOKEN || '';
	if (!discordToken) {
		throw new Error('DISCORD_TOKEN env var is required (do not store tokens in config files)');
	}

	// Precedence: CLI args > config.toml > defaults
	// Precedence: CLI args > config.toml > defaults
	const aiServiceUrl =
		(parsed.aiServiceUrl as string | undefined) ||
		fileCfg.aiServiceUrl ||
		'ws://localhost:8771';
	const chunkMs = fileCfg.chunkMs || 1000;
	const allowedCommanders =
		(parsed.allowedCommanders as string[] | undefined) || fileCfg.allowedCommanders || [];

	// Compute frequently-used absolute paths once
	const logsPath = paths.resolveRoot('logs');
	const wrapupsPath = paths.resolveRoot('wrapups');

	return { discordToken, aiServiceUrl, chunkMs, allowedCommanders, logsPath, wrapupsPath };
}

export function newSessionId(): string {
	return randomUUID();
}
