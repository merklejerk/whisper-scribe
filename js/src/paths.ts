import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

function projectRoot(): string {
	return path.resolve(__dirname, '..', '..');
}

function resolveRoot(...segments: string[]): string {
	return path.join(projectRoot(), ...segments);
}

function ensureDir(dirPath: string): void {
	if (!fs.existsSync(dirPath)) {
		fs.mkdirSync(dirPath, { recursive: true });
	}
}


// New unified data directory helpers
function dataRoot(): string {
    return resolveRoot('data');
}

function sessionDataDir(sessionName: string): string {
    return path.join(dataRoot(), sessionName);
}

function sessionLogPath(sessionName: string): string {
    return path.join(sessionDataDir(sessionName), 'log.jsonl');
}

function sessionWrapupPath(sessionName: string): string {
    return path.join(sessionDataDir(sessionName), 'wrapup.md');
}

export default {
	projectRoot,
	resolveRoot,
	ensureDir,
	dataRoot,
	sessionDataDir,
	sessionLogPath,
	sessionWrapupPath,
};
