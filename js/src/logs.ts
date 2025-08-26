import fs from 'fs';
import path from 'path';

export interface JsonlLogEntry {
	user_id: string;
	user_name: string;
	start_ts: number; // epoch seconds
	end_ts?: number;
	text: string;
	// allow extras without enforcing a schema here
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	[k: string]: any;
}

/**
 * Reads a JSONL log file. Tolerates a single malformed trailing line (e.g., crash mid-write).
 * Throws on any malformed line that is not the final non-empty line.
 */
export async function readLogEntriesStrict(filePath: string): Promise<JsonlLogEntry[]> {
	const out: JsonlLogEntry[] = [];
	if (!fs.existsSync(filePath)) return out;

	const stream = fs.createReadStream(filePath, { encoding: 'utf8' });
	const readline = await import('readline');
	const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

	let lineNo = 0;
	let lastNonEmptyLineNo = 0;
	let malformedLineNo: number | null = null;

	for await (const line of rl as any) {
		lineNo += 1;
		const s = String(line).trim();
		if (!s) continue;
		lastNonEmptyLineNo = lineNo;
		try {
			const obj = JSON.parse(s) as JsonlLogEntry;
			out.push(obj);
		} catch {
			if (malformedLineNo === null) malformedLineNo = lineNo;
		}
	}
	rl.close();

	if (malformedLineNo !== null && malformedLineNo !== lastNonEmptyLineNo) {
		throw new Error(
			`Malformed JSONL at line ${malformedLineNo} in ${path.basename(filePath)} (not trailing)`,
		);
	}

	return out;
}

export function formatLogLines(entries: JsonlLogEntry[]): string {
	return entries
		.map((entry) => {
			const ts = new Date(entry.start_ts * 1000).toLocaleTimeString('en-US', {
				hour12: false,
			});
			return `[${ts}] ${entry.user_name}: ${entry.text}`;
		})
		.join('\n');
}
