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
	yargs(hideBin(process.argv))
		.command(
			'bot <voice-channel-id>',
			'Run Discord STT bot',
			(yargs) => {
				return yargs
					.positional('voice-channel-id', {
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
						alias: 'c',
						type: 'array',
						string: true,
						describe: 'List of user IDs or tags allowed to run commands',
					});
			},
			async (argv) => {
				await runBot({
					voiceChannelId: argv.voiceChannelId,
					aiServiceUrl: argv.aiServiceUrl,
					sessionName: argv.sessionName,
					allowedCommanders: argv.allowedCommanders,
				});
			},
		)
		.command(
			'log',
			'Print formatted log for a session',
			(yargs) => {
				return yargs.positional('session', {
					describe: 'Session name to display log for',
					type: 'string',
					demandOption: true,
				});
			},
			async (argv) => {
				await showLog(argv.session);
			},
		)
		.demandCommand(1, 'You need at least one command before moving on')
		.help()
		.alias('help', 'h').argv;
}

main().catch((e) => {
	console.error('Application error:', e);
	process.exit(1);
});
