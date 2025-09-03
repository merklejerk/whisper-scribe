// Keep in sync with Python side Pydantic models later.
import { z } from 'zod';
import type { JsonlLogEntry } from './logs.js';

export interface BaseMessage {
	v: 1;
	type: string;
}

export interface AudioSegmentMessage extends BaseMessage {
	type: 'audio.segment';
	id: string;
	index: number;
	pcm_format: { sr: number; channels: number; sample_width: number };
	started_ts: number;
	capture_ts: number;
	data_b64: string; // base64 encoded PCM16 LE mono frames
	// Optional per-job Whisper prompt override
	prompt?: string;
}

export interface TranscriptionMessage extends BaseMessage {
	type: 'transcription';
	id: string;
	text: string;
	capture_ts: number;
	end_ts: number;
}

export interface WrapupResponseMessage extends BaseMessage {
	type: 'wrapup.response';
	outline: string;
	request_id: string;
}

export interface ErrorMessage extends BaseMessage {
	type: 'error';
	code: string;
	message: string;
	details?: any;
}

export type InboundFromPython = TranscriptionMessage | ErrorMessage | WrapupResponseMessage;

export interface WrapupRequestMessage extends BaseMessage {
	type: 'wrapup.request';
	session_name: string;
	start_ts: number;
	log_entries: WrapupLogEntry[];
	request_id: string;
	// Optional wrapup overrides
	wrapup_prompt?: string;
	wrapup_tips?: string[];
}

export type OutboundToPython = AudioSegmentMessage | WrapupRequestMessage;

// Wire shape sent to Python summarizer (snake_case, minimal fields)
export interface WrapupLogEntry {
	user_name: string;
	start_ts: number;
	end_ts: number;
	text: string;
	user_id: string;
}

// Convert stored JSONL entries (camelCase internal) into wire shape for wrapup,
// applying optional userid->alias mapping and phrase substitutions.
export function toWrapupLogEntry(
	entry: JsonlLogEntry,
	userIdMap?: Record<string, string>,
	phraseMap?: Record<string, string>,
): WrapupLogEntry {
	const alias = userIdMap?.[entry.userId];
	let text = entry.text;
	if (phraseMap && Object.keys(phraseMap).length) {
		for (const [src, dst] of Object.entries(phraseMap)) {
			if (!src) continue;
			text = text.replaceAll(src, dst);
		}
	}
	return {
		user_name: alias ?? entry.displayName,
		start_ts: entry.startTs,
		end_ts: entry.endTs,
		text,
		user_id: entry.userId,
	};
}

// --- Runtime validation schemas (subset) ---
const base = z.object({ v: z.number(), type: z.string() });
export const transcriptionSchema = base.extend({
	type: z.literal('transcription'),
	id: z.string(),
	text: z.string(),
	capture_ts: z.number(),
	end_ts: z.number(),
});

export const wrapupResponseSchema = base.extend({
	type: z.literal('wrapup.response'),
	outline: z.string(),
	request_id: z.string(),
});

export const errorSchema = base.extend({
	type: z.literal('error'),
	code: z.string(),
	message: z.string(),
	details: z.any().optional(),
});

export const inboundSchema = z.union([transcriptionSchema, wrapupResponseSchema, errorSchema]);
