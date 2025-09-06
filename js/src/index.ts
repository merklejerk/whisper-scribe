import yargs from 'yargs';
import { hideBin } from 'yargs/helpers';
import paths from './paths.js';
import { readLogEntriesStrict, formatLogLines } from './logs.js';
import { runBot } from './bot.js';
import { createWrapup } from './wrapup.js';
import { loadConfig } from './config.js';
import { createSessionGist } from './gist.js';

async function showLog(sessionName: string) {
	const logFilePath = paths.sessionLogPath(sessionName);
	const entries = await readLogEntriesStrict(logFilePath);
	const text = formatLogLines(entries);
	console.log(text || '(empty)');
}

async function generateWrapup(
	sessionName: string,
	forceNew?: boolean,
	profile?: string,
	prevSessionName?: string,
	gist?: boolean,
) {
	const appCfg = loadConfig({ aiServiceUrl: undefined, profile });

	const apiKey = appCfg.geminiApiKey || process.env.GEMINI_API_KEY;
	if (!apiKey) {
		console.error('GEMINI_API_KEY must be set in environment to run wrapup.');
		process.exit(1);
	}

	try {
		const outline = await createWrapup({
			sessionName,
			apiKey,
			userIdMap: appCfg.userIdMap,
			phraseMap: appCfg.phraseMap,
			model: appCfg.wrapupModel,
			prompt: appCfg.wrapupPrompt,
			temperature: appCfg.wrapupTemperature,
			maxOutputTokens: appCfg.wrapupMaxTokens,
			tips: appCfg.wrapupTips,
			forceNew,
			prevSessionName,
		});

		// Always written by createWrapup; CLI prints to stdout
		console.log(outline);

		if (gist) {
			const url = await createSessionGist(sessionName, {
				token: appCfg.githubToken || process.env.GITHUB_TOKEN,
			});
			console.log(`Gist created: ${url}`);
		}
	} catch (err: any) {
		if (err.message !== 'No log entries found for session.') {
			throw err;
		}
		console.error('No log found for that session.');
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
					.option('profile', {
						alias: 'p',
						type: 'string',
						describe: 'Optional profile name to apply config overrides',
					})
					.option('session-name', {
						alias: 's',
						type: 'string',
						describe: 'Session name for logging',
					})
					.option('prev-session', {
						alias: 'P',
						type: 'string',
						describe:
							'Optional previous session name whose wrapup should be used as context',
					})
					.option('gist', {
						type: 'boolean',
						describe: 'Enable uploading wrapup/log to a private GitHub gist on !wrapup',
						default: false,
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
					profile: argv.profile as string | undefined,
					prevSessionName: argv.prevSession as string | undefined,
					gist: (argv.gist as boolean | undefined) || false,
				});
			},
		)
		.command(
			'log <session>',
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
			'Generate or print a wrapup for a recorded session (cached under data/<session>/wrapup.md)',
			(yargs) => {
				return yargs
					.positional('session', {
						describe: 'Session name to generate wrapup for',
						type: 'string',
						demandOption: true,
					})
					.option('profile', {
						alias: 'p',
						type: 'string',
						describe: 'Optional profile name to apply config overrides',
					})
					.option('new', {
						alias: 'n',
						type: 'boolean',
						describe: 'Force generate a new wrapup ignoring the cached version',
						default: false,
					})
					.option('prev-session', {
						alias: 'P',
						type: 'string',
						describe:
							'Optional previous session name whose wrapup should be used as context',
					})
					.option('gist', {
						type: 'boolean',
						describe: 'Upload wrapup.md and log.jsonl to a private GitHub gist',
						default: false,
					});
			},
			async (argv) => {
				await generateWrapup(
					argv.session,
					argv.new as boolean | undefined,
					argv.profile as string | undefined,
					argv.prevSession as string | undefined,
					argv.gist as boolean | undefined,
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
