import { Client, Events, GatewayIntentBits, ChannelType } from 'discord.js';
import {
	joinVoiceChannel,
	getVoiceConnection,
	VoiceConnectionStatus,
	entersState,
} from '@discordjs/voice';
import { loadConfig, newSessionId } from './config.js';
import { Session } from './session.js';
import { UserDirectory } from './userDirectory.js';
import { attachVoiceReceiver } from './voiceReceiver.js';
import { debug } from './debug.js';

export interface RunBotOptions {
	voiceChannelId: string;
	aiServiceUrl?: string;
	sessionName?: string;
	allowedCommanders?: string[];
	profile?: string;
}

export async function runBot(opts: RunBotOptions) {
	const allowedCommanders = opts.allowedCommanders;

	// Load the main app config (does not require voice channel id)
	const cfg = loadConfig({
		aiServiceUrl: opts.aiServiceUrl,
		allowedCommanders,
		profile: opts.profile,
	});

	// Voice channel id MUST be provided via CLI args
	const voiceChannelId = opts.voiceChannelId;
	debug('Configuration loaded:', cfg);
	const client = new Client({
		intents: [
			GatewayIntentBits.Guilds,
			GatewayIntentBits.GuildVoiceStates,
			GatewayIntentBits.GuildMessages,
			GatewayIntentBits.MessageContent,
		],
	});

	let session: Session | null = null;
	let activeGuildId: string | null = null;

	client.once(Events.ClientReady, async () => {
		console.log(`Logged in as ${client.user?.tag}`);
		debug(`Client ready, user: ${client.user?.tag}`);

		try {
			debug(`Attempting to fetch voice channel: ${voiceChannelId}`);
			const voiceChannel = await client.channels.fetch(voiceChannelId as string);
			if (!voiceChannel || !voiceChannel.isVoiceBased()) {
				console.error(
					`Error: Channel ${voiceChannelId} is not a voice channel or could not be found.`,
				);
				process.exit(1);
			}
			debug(`Successfully fetched voice channel: ${voiceChannel.name}`);

			activeGuildId = voiceChannel.guild.id;

			const sessionId = opts.sessionName || newSessionId();
			const userDirectory = new UserDirectory(client);
			session = new Session({
				sessionId,
				guildId: voiceChannel.guild.id,
				voiceChannelId: voiceChannel.id,
				sessionName: opts.sessionName || sessionId,
				aiServiceUrl: cfg.aiServiceUrl,
				logsPath: cfg.logsPath,
				wrapupsPath: cfg.wrapupsPath,
				userDirectory,
				profile: cfg.profile,
				userIdMap: cfg.userIdMap,
				phraseMap: cfg.phraseMap,
				wrapupPrompt: cfg.wrapupPrompt,
				wrapupTips: cfg.wrapupTips,
				asrPrompt: cfg.asrPrompt,
				vadDbThreshold: cfg.vadDbThreshold,
				silenceGapMs: cfg.silenceGapMs,
				vadFrameMs: cfg.vadFrameMs,
				maxSegmentMs: cfg.maxSegmentMs,
			});

			const connection = joinVoiceChannel({
				channelId: voiceChannel.id,
				guildId: voiceChannel.guild.id,
				adapterCreator: voiceChannel.guild.voiceAdapterCreator,
				selfDeaf: false,
				selfMute: false,
			});
			debug('Joining voice channel:', {
				channelId: voiceChannel.id,
				guildId: voiceChannel.guild.id,
			});

			connection.on('stateChange', (oldState, newState) => {
				if (oldState.status !== newState.status) {
					debug(
						`[voice] connection transitioned from ${oldState.status} to ${newState.status}`,
					);
					console.log(
						`[voice] connection transitioned from ${oldState.status} to ${newState.status}`,
					);
				}
			});

			connection.on(VoiceConnectionStatus.Disconnected, async () => {
				try {
					debug('[voice] connection disconnected, attempting to reconnect...');
					await Promise.race([
						entersState(connection, VoiceConnectionStatus.Signalling, 5_000),
						entersState(connection, VoiceConnectionStatus.Connecting, 5_000),
					]);
					// Connection is reconnecting
				} catch (error) {
					console.log('[voice] connection permanently disconnected.');
					debug('[voice] failed to reconnect within 5 seconds, destroying connection.');
					connection.destroy();
				}
			});

			await attachVoiceReceiver(connection, session.getSegmenter());
			session.start();
			console.log(`Successfully joined voice channel: ${voiceChannel.name}`);
			debug('Session started and voice receiver attached.');
		} catch (e) {
			console.error('Failed to join voice channel or start session:', e);
			process.exit(1);
		}
	});

	client.on(Events.MessageCreate, async (message) => {
		if (message.author.bot || !session || message.channel.id !== voiceChannelId) {
			return;
		}
		const contentPreview =
			message.content.length > 50
				? `${message.content.substring(0, 50)}...`
				: message.content;
		debug(`Message received from ${message.author.tag}: "${contentPreview}"`);

		const content = message.content.trim();

		// If it's not a command, treat it as a text message to be logged
		if (!content.startsWith('!')) {
			await session.logTextMessage(message);
			return; // Done with this message
		}

		// It's a command, check if it's one we know
		const command = content;
		if (command !== '!wrapup' && command !== '!log') {
			// Unknown command, ignore it
			debug(`Ignoring unknown command: ${command}`);
			return;
		}

		// If allowedCommanders is non-empty, only allow users listed there.
		if (cfg.allowedCommanders && cfg.allowedCommanders.length > 0) {
			const allowed = cfg.allowedCommanders;
			const authorId = message.author.id;
			const authorTag = message.author.tag; // username#discriminator
			if (!allowed.includes(authorId) && !allowed.includes(authorTag)) {
				// Not allowed; ignore or optionally reply
				debug(
					`User ${authorTag} (${authorId}) is not in the allowed list. Ignoring command.`,
				);
				try {
					await message.reply('You are not authorized to run this command.');
				} catch (e) {
					console.error('Failed to send unauthorized reply:', e);
				}
				return;
			}
		}

		try {
			if (command === '!wrapup') {
				console.log(`Wrapup command received from ${message.author.tag}`);
				await session.handleWrapup(message);
			} else if (command === '!log') {
				console.log(`Log command received from ${message.author.tag}`);
				await session.handleLogCommand(message);
			}
		} catch (e) {
			console.error(`Error processing ${command} command:`, e);
			message.reply(`Sorry, there was an error trying to process the ${command} command.`);
		}
	});

	async function gracefulShutdown() {
		console.log('Shutting down...');
		debug('Graceful shutdown initiated.');
		if (session && activeGuildId) {
			session.stop();
			const connection = getVoiceConnection(activeGuildId);
			connection?.destroy();
			session = null;
		}
		await client.destroy();
		debug('Client destroyed.');
		process.exit(0);
	}

	process.on('SIGINT', gracefulShutdown);
	process.on('SIGTERM', gracefulShutdown);

	await client.login(cfg.discordToken);
	debug('Client logged in.');
}
