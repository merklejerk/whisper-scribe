import json
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description="Convert and sort NDJSON log to text format.")
parser.add_argument("input", help="Input NDJSON log file")
parser.add_argument("output", help="Output text file")
args = parser.parse_args()

entries = []
with open(args.input, "r", encoding="utf-8") as infile:
    for line in infile:
        try:
            obj = json.loads(line)
            entries.append(obj)
        except Exception:
            continue

entries.sort(key=lambda x: x.get("timestamp", ""))

with open(args.output, "w", encoding="utf-8") as outfile:
    for entry in entries:
        name = entry.get("user_name", "?")
        content = entry.get("content", "")
        outfile.write(f"{name}: {content}\n")
