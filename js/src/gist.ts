import fs from 'fs';
import path from 'path';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines } from './logs.js';

export interface GistFileMap {
	[filename: string]: string;
}

export interface GistCreateRequest {
	description?: string;
	public?: boolean;
	files: GistFileMap;
	token?: string;
}

export async function createGist(req: GistCreateRequest): Promise<string> {
	const apiUrl = 'https://api.github.com/gists';
	const headers: Record<string, string> = {
		Accept: 'application/vnd.github+json',
		'Content-Type': 'application/json',
	};
	if (req.token) headers.Authorization = `token ${req.token}`;

	const body = {
		description: req.description ?? '',
		public: req.public ?? false,
		files: Object.fromEntries(
			Object.entries(req.files).map(([fn, content]) => [fn, { content }]),
		),
	};

	const resp = await fetch(apiUrl, { method: 'POST', headers, body: JSON.stringify(body) });
	if (resp.status !== 201) {
		const txt = await resp.text();
		throw new Error(`Failed to create gist: HTTP ${resp.status} ${txt}`);
	}
	const data = await resp.json();
	const url = data?.html_url as string | undefined;
	if (!url) throw new Error('Gist created but no URL returned');
	return url;
}

export function readFilesAsMap(paths: string[]): GistFileMap {
	const files: GistFileMap = {};
	for (const p of paths) {
		const content = fs.readFileSync(p, 'utf8');
		files[path.basename(p)] = content;
	}
	return files;
}

export interface SessionGistOptions {
	token?: string;
	description?: string;
}

export async function createSessionGist(
	sessionName: string,
	opts: SessionGistOptions = {},
): Promise<string> {
	const wrapupPath = paths.sessionWrapupPath(sessionName);
	const logPath = paths.sessionLogPath(sessionName);

	if (!fs.existsSync(wrapupPath)) {
		throw new Error(`Wrapup file not found for session ${sessionName}`);
	}

	const files: GistFileMap = {};
	files['1-wrapup.md'] = fs.readFileSync(wrapupPath, 'utf8');

	const entries = await readLogEntriesStrict(logPath);
	const logTxt = formatLogLines(entries);
	files['2-session-log.txt'] = logTxt;

	return await createGist({
		files,
		description: opts.description ?? `Session ${sessionName}`,
		public: false,
		token: opts.token,
	});
}
