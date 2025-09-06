// A thin, typed wrapper around the CommonJS 'node-vad' package to make it TS-friendly.
// It exposes enums and a small class with convenient methods for Int16Array frames.

import VADImpl from 'node-vad';

export type SampleRate = 8000 | 16000 | 32000 | 48000;

// Mirror the runtime numeric values from node-vad (see lib/vad.js)
export enum VADEvent {
	ERROR = -1,
	SILENCE = 0,
	VOICE = 1,
	NOISE = 2,
}

export enum VADMode {
	NORMAL = 0,
	LOW_BITRATE = 1,
	AGGRESSIVE = 2,
	VERY_AGGRESSIVE = 3,
}

export class WebRtcVad {
	private readonly impl: VADImpl;

	constructor(mode: VADMode = VADMode.AGGRESSIVE) {
		// node-vad expects a number 0..3. Our enum maps 1:1 to those values.
		this.impl = new VADImpl(mode as unknown as number);
	}

	// Accepts mono Int16 PCM frames and returns the VADEvent decision.
	async processFrame(frame: Int16Array, sampleRate: SampleRate): Promise<VADEvent> {
		const buf = Buffer.from(frame.buffer, frame.byteOffset, frame.byteLength);
		const evt = await this.impl.processAudio(buf, sampleRate);
		return evt as unknown as VADEvent;
	}

	// Convenience predicate
	async isVoice(frame: Int16Array, sampleRate: SampleRate): Promise<boolean> {
		return (await this.processFrame(frame, sampleRate)) === VADEvent.VOICE;
	}

	// Expose the underlying instance if low-level access is ever needed.
	get raw(): VADImpl {
		return this.impl;
	}
}
