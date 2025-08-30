// Protocol message type definitions (Phase 1 draft)
// Keep in sync with Python side Pydantic models later.
import { z } from 'zod';
import type { JsonlLogEntry } from './logs.js';

export interface BaseMessage {
	v: 1;
	type: string;
}

export interface AudioChunkMessage extends BaseMessage {
	type: 'audio.chunk';
	user_id: string;
	index: number;
	pcm_format: { sr: number; channels: number; sample_width: number };
	started_ts: number;
	capture_ts: number;
	data_b64: string; // base64 encoded PCM16 LE mono frames
}

export interface TranscriptionMessage extends BaseMessage {
	type: 'transcription';
	user_id: string;
	text: string;
	capture_ts: number;
	end_ts: number;
}

export interface SummarizeResponseMessage extends BaseMessage {
	type: 'summarize.response';
	transcript: string;
	outline: string;
	request_id: string;
}

export interface ErrorMessage extends BaseMessage {
	type: 'error';
	code: string;
	message: string;
	details?: any;
}

export type InboundFromPython = TranscriptionMessage | ErrorMessage | SummarizeResponseMessage;

export interface WrapupRequestMessage extends BaseMessage {
	type: 'summarize.request';
	session_name: string;
	start_ts: number;
	log_entries: WrapupLogEntry[];
	request_id: string;
}

export type OutboundToPython = AudioChunkMessage | WrapupRequestMessage;

// Wire shape sent to Python summarizer (snake_case, minimal fields)
export interface WrapupLogEntry {
	user_name: string;
	start_ts: number;
	end_ts: number;
	text: string;
	user_id: string;
}

// Convert stored JSONL entries (camelCase internal) into wire shape for wrapup
export function toWrapupLogEntry(entry: JsonlLogEntry): WrapupLogEntry {
	return {
		user_name: entry.displayName,
		start_ts: entry.startTs,
		end_ts: entry.endTs,
		text: entry.text,
		user_id: entry.userId,
	};
}

// --- Runtime validation schemas (subset) ---
const base = z.object({ v: z.number(), type: z.string() });
export const transcriptionSchema = base.extend({
	type: z.literal('transcription'),
	user_id: z.string(),
	text: z.string(),
	capture_ts: z.number(),
	end_ts: z.number(),
});

export const summarizeResponseSchema = base.extend({
	type: z.literal('summarize.response'),
	transcript: z.string(),
	outline: z.string(),
	request_id: z.string(),
});

export const errorSchema = base.extend({
	type: z.literal('error'),
	code: z.string(),
	message: z.string(),
	details: z.any().optional(),
});

export const inboundSchema = z.union([transcriptionSchema, summarizeResponseSchema, errorSchema]);
