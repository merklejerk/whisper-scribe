import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import fs from 'fs';
import path from 'path';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines } from './logs.js';
import { runBot } from './bot.js';

async function showLog(sessionName: string) {
	const logsDir = paths.resolveRoot('logs');
	const logFilePath = path.join(logsDir, `${sessionName}.jsonl`);

	if (!fs.existsSync(logFilePath)) {
		console.error(`No log found for session: ${sessionName}`);
		console.error(`Expected file: ${logFilePath}`);
		process.exit(1);
	}

	console.log(`Session Log: ${sessionName}`);
	console.log('='.repeat(50));

	try {
		const entries = await readLogEntriesStrict(logFilePath);
		const text = formatLogLines(entries);
		console.log(text || '(empty)');
	} catch (e) {
		console.error('Failed to read session log:', e);
		process.exit(1);
	}
}

async function main() {
	const argv = yargs(hideBin(process.argv))
		.command('bot <voice>', 'Run Discord STT bot', (yargs) => {
			return yargs
				.positional('voice', {
					describe: 'Voice channel ID to join',
					type: 'string',
					demandOption: true,
				})
				.option('ai-service-url', {
					alias: 'a',
					type: 'string',
					describe: 'AI service URL',
				})
				.option('session-name', {
					alias: 's',
					type: 'string',
					describe: 'Session name for logging',
				})
				.option('allowed-commanders', {
					alias: 'a',
					type: 'array',
					describe: 'User IDs or tags allowed to run commands (repeatable)',
				});
		})
		.command('log <session>', 'Print formatted log for a session', (yargs) => {
			return yargs.positional('session', {
				describe: 'Session name to display log for',
				type: 'string',
				demandOption: true,
			});
		})
		.demandCommand(1, 'You must specify a command')
		.help()
		.parseSync();

	const command = argv._[0];

	if (command === 'bot') {
		await runBot(argv);
	} else if (command === 'log') {
		await showLog(argv.session as string);
	} else {
		console.error(`Unknown command: ${command}`);
		process.exit(1);
	}
}

main().catch((err) => {
	console.error('Application error:', err);
	process.exit(1);
});
