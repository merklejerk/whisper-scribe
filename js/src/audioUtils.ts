// Audio utility helpers for PCM manipulation and resampling.
// Keep dependency‑free (no native addons) for portability.

export interface PcmChunkMeta {
	sampleRate: number;
	channels: number;
	sampleWidth: number; // bytes per sample
}

// Downmix interleaved Int16 stereo/mono data to mono Int16.
export function downmixToMono(int16: Int16Array, channels: number): Int16Array {
	if (channels === 1) return int16;
	if (channels !== 2) {
		throw new Error(`Unsupported channel count: ${channels}`);
	}
	const out = new Int16Array(int16.length / 2);
	for (let i = 0, j = 0; i < int16.length; i += 2, j++) {
		// simple average; keep in range
		const v = (int16[i] + int16[i + 1]) / 2;
		out[j] = v < -32768 ? -32768 : v > 32767 ? 32767 : v;
	}
	return out;
}

// Linear resample Int16 mono PCM to new sample rate. Returns Int16Array.
export function resampleLinear(int16: Int16Array, fromRate: number, toRate: number): Int16Array {
	if (fromRate === toRate) return int16;
	const ratio = toRate / fromRate;
	const newLength = Math.max(1, Math.round(int16.length * ratio));
	const out = new Int16Array(newLength);
	for (let i = 0; i < newLength; i++) {
		const srcPos = i / ratio;
		const i0 = Math.floor(srcPos);
		const i1 = Math.min(int16.length - 1, i0 + 1);
		const t = srcPos - i0;
		out[i] = (int16[i0] * (1 - t) + int16[i1] * t) | 0;
	}
	return out;
}

export function pcm16ToBase64(int16: Int16Array): string {
	return Buffer.from(int16.buffer, int16.byteOffset, int16.byteLength).toString('base64');
}

export function nowIso(): string {
	return new Date().toISOString();
}

export function nowEpoch(): number {
	return Date.now() / 1000;
}

// Compute RMS in dBFS for a mono Int16 frame. Returns negative dB value (0 dBFS is full scale).
export function rmsDbFs(frame: Int16Array): number {
	if (frame.length === 0) return -Infinity;
	let sumSq = 0;
	// Use scalar loop to avoid creating a Float32Array; keep GC low on small frames
	for (let i = 0; i < frame.length; i++) {
		const v = frame[i] / 32768;
		sumSq += v * v;
	}
	const rms = Math.sqrt(sumSq / frame.length);
	const eps = 1e-9;
	return 20 * Math.log10(Math.max(rms, eps));
}
