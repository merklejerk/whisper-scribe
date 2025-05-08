import argparse
import asyncio
from src.bot import DiscordBot
import src.config as config
from src.logging import load_log
from src.wrapup import create_wrapup_from_log_entries

def main():
    parser = argparse.ArgumentParser(description="Discord Bot CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Bot command
    bot_parser = subparsers.add_parser("bot", help="Run the Discord bot as normal.")
    bot_parser.add_argument("voice_channel_id", type=int, help="Voice channel ID to auto-join on startup.")
    bot_parser.add_argument("session_name", type=str, help="Session name to use for log file naming.")
    bot_parser.add_argument("--device", type=str, default="cpu", help="Device for Whisper model (e.g. 'cpu', 'cuda', 'mps')")
    bot_parser.add_argument("--log-metadata", action="store_true", help="Include detailed metadata in log entries.")

    # Wrapup command
    process_parser = subparsers.add_parser("wrapup", help="Process a log file and generate an outline.")
    process_parser.add_argument("logfile", type=str, help="Path to the log file (ndjson)")
    process_parser.add_argument("name", type=str, help="Name to use for output files")

    args = parser.parse_args()

    if args.command == "bot":
        async def run_bot():
            bot = None
            try:
                bot = DiscordBot(
                    voice_channel_id=args.voice_channel_id,
                    session_name=args.session_name,
                    device=args.device,
                    log_metadata=args.log_metadata,
                )
                await bot.start(config.DISCORD_TOKEN)
            finally:
                if bot:
                    await bot.shutdown()
        asyncio.run(run_bot())
    elif args.command == "wrapup":
        async def run_wrapup():
            entries = await load_log(args.logfile)
            await create_wrapup_from_log_entries(entries, args.name)
        asyncio.run(run_wrapup())

if __name__ == "__main__":
    main()
