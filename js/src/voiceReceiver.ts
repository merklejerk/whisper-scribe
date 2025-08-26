import {
	EndBehaviorType,
	VoiceConnection,
	VoiceConnectionStatus,
	entersState,
} from '@discordjs/voice';
import { Client } from 'discord.js';
import prism from 'prism-media';
import { resampleLinear, downmixToMono, nowEpoch } from './audioUtils.js';

// Discord supplies Opus packets; we use the @discordjs/voice receiver which gives us PCM 16-bit LE stereo 48kHz frames.
// We aggregate per-user frames into time-based chunks and invoke a callback with internal VoiceChunk objects.

export interface VoiceReceiverOptions {
	targetSampleRate: number; // e.g. 16000
	chunkMs: number; // aggregation window
	maxBufferedChunks?: number; // backpressure guard (# of chunks not yet flushed upstream)
}

export interface VoiceChunk {
	user_id: string;
	index: number;
	pcm_format: { sr: number; channels: number; sample_width: number };
	started_ts: number;
	capture_ts: number;
	duration_ms: number;
	pcm16: Int16Array; // raw PCM samples at target SR, mono, 16-bit
}

interface UserBuf {
	samples: Int16Array[]; // list of partial PCM blocks (mono, target SR)
	total: number; // total samples accumulated
	nextIndex: number; // sequential chunk index per user
	startedTs?: number; // epoch seconds of first frame in current chunk
}

export class PerUserVoiceChunker {
	private users = new Map<string, UserBuf>();
	// sessionId previously set via setter; now fixed at construction to avoid temporal coupling
	constructor(
		private opts: VoiceReceiverOptions,
		private send: (chunk: VoiceChunk) => void,
	) {}

	pushStereo48(userId: string, pcmStereo48: Int16Array) {
		// Convert: stereo 48k -> mono -> resample to target SR
		const mono48 = downmixToMono(pcmStereo48, 2);
		const monoTgt = resampleLinear(mono48, 48000, this.opts.targetSampleRate);
		let state = this.users.get(userId);
		if (!state) {
			state = { samples: [], total: 0, nextIndex: 0 };
			this.users.set(userId, state);
		}
		if (!state.startedTs) state.startedTs = nowEpoch();
		state.samples.push(monoTgt);
		state.total += monoTgt.length;
		// If we've exceeded chunk window (approx by samples)
		const msPerSample = 1000 / this.opts.targetSampleRate;
		const estMs = state.total * msPerSample;
		if (estMs >= this.opts.chunkMs) this.flushUser(userId, state);
	}

	flushAll() {
		for (const [uid, st] of this.users) this.flushUser(uid, st);
	}

	private flushUser(userId: string, state: UserBuf) {
		if (!state.total) return;
		const data = concatInt16(state.samples, state.total);
		const durationMs = Math.round((state.total * 1000) / this.opts.targetSampleRate);
		const chunk: VoiceChunk = {
			user_id: userId,
			index: state.nextIndex++,
			pcm_format: { sr: this.opts.targetSampleRate, channels: 1, sample_width: 2 },
			started_ts: state.startedTs!,
			capture_ts: nowEpoch(),
			duration_ms: durationMs,
			pcm16: data,
		};
		state.samples = [];
		state.total = 0;
		state.startedTs = undefined;
		this.send(chunk);
	}
}

function concatInt16(chunks: Int16Array[], total: number) {
	const out = new Int16Array(total);
	let off = 0;
	for (const c of chunks) {
		out.set(c, off);
		off += c.length;
	}
	return out;
}

// Attach per-user opus decoding + chunk aggregation. Returns a disposer.
export async function attachVoiceReceiver(
	client: Client,
	connection: VoiceConnection,
	chunker: PerUserVoiceChunker,
) {
	await entersState(connection, VoiceConnectionStatus.Ready, 30_000).catch((e) => {
		throw new Error('Voice connection not ready: ' + e);
	});
	const speaking = connection.receiver.speaking;
	const activeDecoders = new Map<string, prism.opus.Decoder>();

	speaking.on('start', (userId: string) => {
		if (activeDecoders.has(userId)) return;
		const opusStream = connection.receiver.subscribe(userId, {
			end: { behavior: EndBehaviorType.AfterSilence, duration: 250 },
		});
		const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
		activeDecoders.set(userId, decoder);
		opusStream.pipe(decoder);
		decoder.on('data', (buf: Buffer) => {
			// buf PCM16LE stereo 48k
			const int16 = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
			chunker.pushStereo48(userId, int16);
		});
		const cleanup = () => {
			decoder.removeAllListeners('data');
			decoder.destroy();
			activeDecoders.delete(userId);
			chunker.flushAll();
		};
		opusStream.once('end', cleanup);
		opusStream.once('close', cleanup);
		opusStream.once('error', () => cleanup());
	});

	speaking.on('end', (userId: string) => {
		chunker.flushAll();
	});

	const flushTimer = setInterval(() => chunker.flushAll(), 1000);
	return () => {
		clearInterval(flushTimer);
		for (const dec of activeDecoders.values()) dec.destroy();
		activeDecoders.clear();
	};
}
