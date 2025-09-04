import {
	EndBehaviorType,
	VoiceConnection,
	VoiceConnectionStatus,
	entersState,
} from '@discordjs/voice';
import prism from 'prism-media';
import { resampleLinear, downmixToMono, nowEpoch, rmsDbFs } from './audioUtils.js';
import { debug } from './debug.js';
import { WebRtcVad, VADMode, VADEvent } from './vad.js';

// Discord supplies Opus packets; we use the @discordjs/voice receiver which gives us PCM 16-bit LE stereo 48kHz frames.
// We aggregate per-user frames (Discord chunks) and emit pre-segmented speech as VoiceSegment objects.

export const TARGET_SR = 16000;

export interface SegmenterOptions {
	// Primitive VAD controls (energy-based)
	vadDbThreshold?: number; // e.g. -45 dBFS
	silenceGapMs?: number; // e.g. 1250 ms
	vadFrameMs?: number; // e.g. 30 ms
	maxSegmentMs?: number; // e.g. 15000 ms
	// Minimum segment duration; avoid emitting tiny blips
	minSegmentMs?: number; // e.g. 200 ms
	// WebRTC VAD aggressiveness mode. Default: AGGRESSIVE.
	webrtcVadMode?: VADMode;
	// Callback invoked when a speech segment is finalized
	send: (segment: VoiceSegment) => void;
}

export interface VoiceSegment {
	userId: string;
	index: number;
	pcmFormat: { sampleRate: number; channels: number; sampleWidth: number };
	startedTs: number;
	captureTs: number;
	durationMs: number;
	pcm16: Int16Array; // raw PCM samples at target SR, mono, 16-bit
}

interface UserBuf {
	frames: Int16Array[]; // frames included in current speech segment
	totalInSeg: number; // total samples in current speech segment
	nextIndex: number; // sequential chunk index per user
	startedTs?: number; // epoch seconds of segment start
	inSpeech: boolean;
	// Track last active (speech) position in samples from seg start
	lastActiveSamples: number;
	// Count of consecutive silence samples since last speech
	silenceSamples: number;
	// Buffer of contiguous silence frames while inSpeech, to stitch if speech resumes
	pendingSilence?: Int16Array[];
	pendingSilenceSamples?: number;
	// Carry-over from previous push to form fixed-size VAD frames
	carry?: Int16Array;
	// Buffered mono@TARGET_SR samples awaiting segmentation
	inQueue: Int16Array[];
	// Per-user node-vad instance (wrapped)
	vad?: WebRtcVad;
}

export class PerUserVoiceSegmenter {
	private users = new Map<string, UserBuf>();
	private _flushing = false;

	constructor(private opts: SegmenterOptions) {
		debug(
			'PerUserVoiceSegmenter created (vadFrameMs=%s, thresh=%s dB, silenceGapMs=%s, minSegmentMs=%s, maxSegmentMs=%s)',
			String(opts.vadFrameMs ?? 30),
			String(opts.vadDbThreshold ?? -45),
			String(opts.silenceGapMs ?? 1250),
			String(opts.minSegmentMs ?? 200),
			String(opts.maxSegmentMs ?? 30000),
		);
	}

	pushStereo48(userId: string, pcmStereo48: Int16Array) {
		// Convert: stereo 48k -> mono -> resample to target SR, then buffer
		const mono48 = downmixToMono(pcmStereo48, 2);
		const monoTgt = resampleLinear(mono48, 48000, TARGET_SR);

		let state = this.users.get(userId);
		if (!state) {
			state = {
				frames: [],
				totalInSeg: 0,
				nextIndex: 0,
				inSpeech: false,
				lastActiveSamples: 0,
				silenceSamples: 0,
				pendingSilence: [],
				pendingSilenceSamples: 0,
				inQueue: [],
			};
			this.users.set(userId, state);
			debug('segmenter: new user state created user=%s', userId);
		}

		state.inQueue.push(monoTgt);
	}

	async flushAll(): Promise<void> {
		// Reentrancy guard to avoid overlapping async flushes
		if (this._flushing) {
			return;
		}
		this._flushing = true;
		const p = this._getSegParams();
		for (const [uid, state] of this.users) {
			const work = this._buildWorkBuffer(state);

			let offset = 0;
			let processed = false;
			if (work) {
				while (work && offset + p.vadFrameSamples <= work.length) {
					const frame = work.subarray(offset, offset + p.vadFrameSamples);
					offset += p.vadFrameSamples;
					processed = true;

					// First pass: simple energy threshold
					const energyActive = rmsDbFs(frame) >= p.thresh;
					let active = false;
					if (energyActive) {
						// Second pass: node-vad (WebRTC) confirmation
						if (!state.vad) {
							const mode = this.opts.webrtcVadMode ?? VADMode.AGGRESSIVE;
							state.vad = new WebRtcVad(mode);
						}
						active = await state.vad.isVoice(frame, TARGET_SR);
					}

					if (!state.inSpeech) {
						if (active) {
							this._startSegment(uid, state);
							this._appendActiveFrame(state, frame);
						}
						continue;
					}

					if (active) {
						// If we had buffered silence during the ongoing segment, stitch it back in
						if (
							state.pendingSilenceSamples &&
							state.pendingSilence &&
							state.pendingSilence.length
						) {
							for (const s of state.pendingSilence) {
								state.frames.push(s.slice());
								state.totalInSeg += s.length;
							}
							state.pendingSilence = [];
							state.pendingSilenceSamples = 0;
							state.silenceSamples = 0;
							state.lastActiveSamples = state.totalInSeg;
						}
						this._appendActiveFrame(state, frame);
					} else {
						// While in an active segment, buffer contiguous silent frames to allow stitching
						if (!state.pendingSilence) state.pendingSilence = [];
						state.pendingSilence.push(frame.slice());
						state.pendingSilenceSamples =
							(state.pendingSilenceSamples || 0) + frame.length;
						state.silenceSamples += frame.length;
					}
				}
				// Save carry (remainder) for next flush
				if (offset < work.length) {
					state.carry = work.subarray(offset).slice();
				}
			}

			// Decide finalization after processing this buffer
			this._maybeFinalizePost(uid, state, p, processed);
		}
		this._flushing = false;
	}

	private _getSegParams() {
		const vadFrameMs = this.opts.vadFrameMs ?? 30;
		return {
			vadFrameSamples: Math.max(1, Math.round((TARGET_SR * vadFrameMs) / 1000)),
			thresh: this.opts.vadDbThreshold ?? -45,
			silenceGapMs: this.opts.silenceGapMs ?? 1250,
			minSegmentMs: this.opts.minSegmentMs ?? 200,
			maxSegmentMs: this.opts.maxSegmentMs ?? 30000,
		};
	}

	private _buildWorkBuffer(state: UserBuf): Int16Array | null {
		const carryLen = state.carry?.length ?? 0;
		let totalLen = carryLen;
		for (const q of state.inQueue) totalLen += q.length;
		if (totalLen === 0) return null;
		const work = new Int16Array(totalLen);
		let woff = 0;
		if (carryLen) {
			work.set(state.carry!, 0);
			woff += carryLen;
		}
		for (const q of state.inQueue) {
			work.set(q, woff);
			woff += q.length;
		}
		// Clear queue & carry; carry will be set from remainder after processing
		state.inQueue = [];
		state.carry = undefined;
		return work;
	}

	private _startSegment(userId: string, state: UserBuf) {
		state.inSpeech = true;
		state.startedTs = nowEpoch();
		state.frames = [];
		state.totalInSeg = 0;
		state.lastActiveSamples = 0;
		state.silenceSamples = 0;
		state.pendingSilence = [];
		state.pendingSilenceSamples = 0;
		debug('segmenter: start segment user=%s ts=%s', userId, String(state.startedTs));
	}

	private _appendActiveFrame(state: UserBuf, frame: Int16Array) {
		state.frames.push(frame.slice());
		state.totalInSeg += frame.length;
		state.lastActiveSamples = state.totalInSeg;
		state.silenceSamples = 0;
	}

	private _maybeFinalizePost(
		uid: string,
		state: UserBuf,
		p: {
			vadFrameSamples: number;
			thresh: number;
			silenceGapMs: number;
			minSegmentMs: number;
			maxSegmentMs: number;
		},
		processed: boolean,
	) {
		if (!state.inSpeech || !state.startedTs || state.totalInSeg <= 0) return;
		// Compute silence based on what we processed this flush; if none, use wall-clock
		let silentMs: number;
		if (processed) {
			silentMs = (state.silenceSamples * 1000) / TARGET_SR;
		} else {
			const lastActiveTs = state.startedTs + state.lastActiveSamples / TARGET_SR;
			const now = nowEpoch();
			silentMs = Math.max(0, (now - lastActiveTs) * 1000);
		}
		// Also compute current duration
		const durMs = (state.totalInSeg * 1000) / TARGET_SR;

		// Enforce minimum segment duration to avoid tiny emissions
		if (durMs < p.minSegmentMs) {
			// Do not finalize yet, even if we've seen a silence gap
			return;
		}

		if (silentMs >= p.silenceGapMs) {
			this._finalizeSegment(uid, state);
			return;
		}
		// Also enforce max segment length centrally
		if (durMs >= p.maxSegmentMs) this._finalizeSegment(uid, state);
	}

	private _resetSpeechState(state: UserBuf) {
		state.frames = [];
		state.totalInSeg = 0;
		state.inSpeech = false;
		state.startedTs = undefined;
		state.lastActiveSamples = 0;
		state.silenceSamples = 0;
		state.pendingSilence = [];
		state.pendingSilenceSamples = 0;
	}

	private _finalizeSegment(userId: string, state: UserBuf) {
		if (!state.inSpeech || !state.startedTs || state.totalInSeg <= 0) {
			// Nothing to emit — just reset speech-related state
			this._resetSpeechState(state);
			return;
		}
		// Trim to last active sample to drop trailing silence
		const keepSamples = Math.max(0, Math.min(state.lastActiveSamples, state.totalInSeg));
		if (keepSamples <= 0) {
			// Reset and return without emitting
			this._resetSpeechState(state);
			return;
		}
		const data = concatInt16Trimmed(state.frames, keepSamples);
		const durationMs = Math.round((keepSamples * 1000) / TARGET_SR);
		const endTs = state.startedTs + keepSamples / TARGET_SR;
		const segment: VoiceSegment = {
			userId,
			index: state.nextIndex++,
			pcmFormat: { sampleRate: TARGET_SR, channels: 1, sampleWidth: 2 },
			startedTs: state.startedTs,
			captureTs: endTs,
			durationMs,
			pcm16: data,
		};
		debug(
			'segmenter: finalize segment user=%s idx=%s durMs=%s',
			userId,
			String(segment.index),
			String(durationMs),
		);
		// Reset state for next segment
		this._resetSpeechState(state);
		// Do not clear carry; it belongs to stream-level framing
		this.opts.send(segment);
	}
}

function concatInt16Trimmed(chunks: Int16Array[], keepSamples: number) {
	const out = new Int16Array(keepSamples);
	let off = 0;
	for (const c of chunks) {
		if (off >= keepSamples) break;
		const remaining = keepSamples - off;
		if (c.length <= remaining) {
			out.set(c, off);
			off += c.length;
		} else {
			out.set(c.subarray(0, remaining), off);
			off += remaining;
		}
	}
	return out;
}

// Attach per-user opus decoding + chunk aggregation. Returns a disposer.
export async function attachVoiceReceiver(
	connection: VoiceConnection,
	segmenter: PerUserVoiceSegmenter,
) {
	await entersState(connection, VoiceConnectionStatus.Ready, 30_000).catch((e) => {
		throw new Error('Voice connection not ready: ' + e);
	});
	const speaking = connection.receiver.speaking;
	const activeDecoders = new Map<string, prism.opus.Decoder>();

	speaking.on('start', (userId: string) => {
		if (activeDecoders.has(userId)) return;
		const opusStream = connection.receiver.subscribe(userId, {
			end: { behavior: EndBehaviorType.AfterInactivity, duration: 250 },
		});
		const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
		activeDecoders.set(userId, decoder);
		opusStream.pipe(decoder);
		decoder.on('data', (buf: Buffer) => {
			// buf PCM16LE stereo 48k
			const int16 = new Int16Array(buf.buffer, buf.byteOffset, buf.byteLength / 2);
			segmenter.pushStereo48(userId, int16);
		});
		const cleanup = () => {
			debug(`Stream closed for user ${userId}`);
			decoder.removeAllListeners('data');
			decoder.destroy();
			activeDecoders.delete(userId);
			void segmenter.flushAll();
		};
		opusStream.once('end', cleanup);
		opusStream.once('close', cleanup);
		opusStream.once('error', () => cleanup());
	});

	speaking.on('end', (userId: string) => {
		void segmenter.flushAll();
	});

	const flushTimer = setInterval(() => void segmenter.flushAll(), 1000);
	return () => {
		clearInterval(flushTimer);
		for (const dec of activeDecoders.values()) dec.destroy();
		activeDecoders.clear();
	};
}
