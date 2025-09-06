import { Client, Guild } from 'discord.js';

// Centralized cache and resolver for user display names, scoped by guild.
export class UserDirectory {
	private client: Client;
	// Cache per guildId -> (userId -> displayName)
	private displayNames = new Map<string, Map<string, string>>();

	constructor(client: Client) {
		this.client = client;
		// Keep cache fresh on member updates
		this.client.on('guildMemberUpdate', (_old, member) => {
			const name = member.displayName;
			this.setDisplayName(member.guild.id, member.id, name);
		});
	}

	getDisplayNameSync(userId: string, guildId?: string): string | undefined {
		if (!guildId) return undefined;
		return this.displayNames.get(guildId)?.get(userId);
	}

	async ensureDisplayName(userId: string, guildId?: string) {
		if (!guildId) return;
		if (this.displayNames.get(guildId)?.has(userId)) return;
		try {
			// Try cached guild
			const guild: Guild | undefined = this.client.guilds.cache.get(guildId) ?? undefined;
			const g = guild ?? (await this.client.guilds.fetch(guildId).catch(() => null));
			if (!g) return;
			const member = await g.members.fetch(userId).catch(() => null);
			const user = member?.user ?? (await this.client.users.fetch(userId).catch(() => null));
			const displayName = member?.displayName ?? (user as any)?.globalName ?? user?.username;
			if (displayName) this.setDisplayName(guildId, userId, displayName);
		} catch {
			// ignore fetch errors; best-effort only
		}
	}

	private setDisplayName(guildId: string, userId: string, displayName: string) {
		let map = this.displayNames.get(guildId);
		if (!map) {
			map = new Map();
			this.displayNames.set(guildId, map);
		}
		map.set(userId, displayName);
	}
}
