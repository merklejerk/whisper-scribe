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

function logsPath(filename: string): string {
	return resolveRoot('logs', filename);
}

export default {
	projectRoot,
	resolveRoot,
	ensureDir,
	logsPath,
};
