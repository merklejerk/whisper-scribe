import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import path from 'path';
import fs from 'fs';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines } from './logs.js';
import { runBot } from './bot.js';
import { PythonWsClient } from './websocketClient.js';
import { toWrapupLogEntry, WrapupRequestMessage } from './messages.js';
import { randomUUID } from 'crypto';
import { loadConfig } from './config.js';

async function showLog(sessionName: string) {
	const logsDir = loadConfig({}).logsPath; // will throw if DISCORD_TOKEN missing; acceptable for CLI
	const logFilePath = path.join(logsDir, `${sessionName}.jsonl`);
	const entries = await readLogEntriesStrict(logFilePath);
	const text = formatLogLines(entries);
	console.log(text || '(empty)');
}

async function generateWrapup(sessionName: string, aiServiceUrl?: string, writeOut?: boolean) {
	const appCfg = loadConfig({ aiServiceUrl });
	const logsDir = appCfg.logsPath;
	const logFilePath = path.join(logsDir, `${sessionName}.jsonl`);
	const entries = await readLogEntriesStrict(logFilePath);
	const wrapupEntries = entries.map(toWrapupLogEntry);
	const requestId = randomUUID();

	let clientResolve: ((v: any) => void) | null = null;
	let clientReject: ((e: any) => void) | null = null;

	const respPromise = new Promise((resolve, reject) => {
		clientResolve = resolve;
		clientReject = reject;
		// timeout after 90s
		setTimeout(() => reject(new Error('Timed out waiting for wrapup response')), 90_000);
	});

	const client = new PythonWsClient(appCfg.aiServiceUrl, (msg) => {
		if (msg.type === 'wrapup.response' && (msg as any).request_id === requestId) {
			if (clientResolve) clientResolve(msg);
		}
		if (msg.type === 'error') {
			if (clientReject) clientReject(new Error(`Remote error: ${JSON.stringify(msg)}`));
		}
	});

	client.setOnOpen(() => {
		// Use the timestamp of the first log entry if available, otherwise use now
		const firstEntryTs =
			entries && entries.length > 0 && entries[0].startTs
				? entries[0].startTs
				: Date.now() / 1000;
		const wrapupReq: WrapupRequestMessage = {
			v: 1,
			type: 'wrapup.request',
			session_name: sessionName,
			start_ts: firstEntryTs,
			log_entries: wrapupEntries,
			request_id: requestId,
		};
		client.send(wrapupReq);
	});

	client.start();

	let resp: any;
	try {
		resp = await respPromise;
	} catch (e) {
		console.error('Failed to get wrapup response:', e);
		process.exit(1);
	} finally {
		client.stop();
	}

	if (writeOut) {
		// Ensure wrapups dir exists
		const wrapupsDir = appCfg.wrapupsPath;
		paths.ensureDir(wrapupsDir);
		const outPath = path.join(wrapupsDir, `${sessionName}.md`);
		fs.writeFileSync(outPath, resp.outline, 'utf8');
		console.log(`Wrote wrapup to: ${outPath}`);
	} else {
		console.log(resp.outline);
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
		.command(
			'wrapup <session>',
			'Generate wrapup files for a recorded session (writes to wrapups/)',
			(yargs) => {
				return yargs
					.positional('session', {
						describe: 'Session name to generate wrapup for',
						type: 'string',
						demandOption: true,
					})
					.option('ai-service-url', {
						alias: 'a',
						type: 'string',
						describe: 'AI service websocket URL (overrides config)',
					})
					.option('output', {
						alias: 'O',
						type: 'boolean',
						describe:
							'Write wrapup to wrapups/<session>.md instead of printing to stdout',
					});
			},
			async (argv) => {
				await generateWrapup(
					argv.session,
					argv['ai-service-url'] as string | undefined,
					argv.output as boolean | undefined,
				);
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
