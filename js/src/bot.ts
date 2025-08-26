import { Client, Events, GatewayIntentBits, ChannelType } from 'discord.js';
import {
	joinVoiceChannel,
	getVoiceConnection,
	VoiceConnectionStatus,
	entersState,
} from '@discordjs/voice';
import { loadConfig, newSessionId } from './config.js';
import { Session } from './session.js';
import { attachVoiceReceiver } from './voiceReceiver.js';
import { debug } from './debug.js';

export async function runBot(argv: any) {
	// First positional arg is the required voice channel id
	const voicePos = argv.voice;

	let allowedCommanders: string[] | undefined;
	if (argv['allowed-commanders']) {
		const raw = argv['allowed-commanders'] as unknown as Array<unknown>;
		allowedCommanders = raw.map((x) => String(x).trim());
	} else {
		allowedCommanders = undefined;
	}

	const cfg = loadConfig({
		voice: voicePos,
		aiServiceUrl: argv['ai-service-url'],
		sessionName: argv['session-name'],
		allowedCommanders,
	});
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

		if (!cfg.voiceChannelId) {
			console.error(
				'Error: Voice channel ID is required. Please provide it via config file or --voice argument.',
			);
			process.exit(1);
		}

		try {
			debug(`Attempting to fetch voice channel: ${cfg.voiceChannelId}`);
			const voiceChannel = await client.channels.fetch(cfg.voiceChannelId);
			if (!voiceChannel || !voiceChannel.isVoiceBased()) {
				console.error(
					`Error: Channel ${cfg.voiceChannelId} is not a voice channel or could not be found.`,
				);
				process.exit(1);
			}
			debug(`Successfully fetched voice channel: ${voiceChannel.name}`);

			activeGuildId = voiceChannel.guild.id;

			const sessionId = cfg.sessionName || newSessionId();
			session = new Session({
				sessionId,
				voiceChannelId: voiceChannel.id,
				startTimestamp: Date.now() / 1000,
				sessionName: cfg.sessionName || sessionId,
				aiServiceUrl: cfg.aiServiceUrl,
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

			await attachVoiceReceiver(client, connection, session.getChunker());
			session.start();
			console.log(`Successfully joined voice channel: ${voiceChannel.name}`);
			debug('Session started and voice receiver attached.');
		} catch (e) {
			console.error('Failed to join voice channel or start session:', e);
			process.exit(1);
		}
	});

	client.on(Events.MessageCreate, async (message) => {
		if (message.author.bot || !session) {
			return;
		}
		const contentPreview =
			message.content.length > 50 ? `${message.content.substring(0, 50)}...` : message.content;
		debug(`Message received from ${message.author.tag}: "${contentPreview}"`);

		const command = message.content.trim();
		if (command !== '!wrapup' && command !== '!log') {
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
				console.log(
					`User ${authorTag} (${authorId}) attempted to run ${command} but is not permitted.`,
				);
				try {
					await message.reply('You are not authorized to run this command.');
				} catch (e) {
					console.error('Failed to send unauthorized reply:', e);
				}
				return;
			}
			debug(`User ${authorTag} (${authorId}) is authorized.`);
		}

		try {
			const voiceChannel = await client.channels.fetch(session.getVoiceChannelId());
			if (!voiceChannel || !voiceChannel.isVoiceBased()) {
				return; // Should not happen if session is active
			}
			debug(`Command invoked in channel: ${message.channel.id}, voice channel: ${voiceChannel.id}`);

			if (
				message.channel.type === ChannelType.GuildText &&
				message.channel.name === voiceChannel.name
			) {
				if (command === '!wrapup') {
					console.log(
						`Wrapup command received from ${message.author.tag} in #${message.channel.name}`,
					);
					await session.handleWrapup(message);
				} else if (command === '!log') {
					console.log(
						`Log command received from ${message.author.tag} in #${message.channel.name}`,
					);
					await session.handleLogCommand(message);
				}
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
