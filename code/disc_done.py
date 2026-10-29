"""Exit 0 once every discovery prompt's 64 Stage A samples are judged (thinking-off)."""
import json
import sys

from common import DATA, RES

disc = {k for k, v in json.loads((DATA / "split.json").read_text()).items() if v["split"] == "discovery"}
done = {json.loads(l)["key"] for l in open(RES / f"stage_a_judge_{sys.argv[1] if len(sys.argv) > 1 else 'off'}.jsonl")}
miss = len({f"{p}|{i}" for p in disc for i in range(64)} - done)
print(miss)
sys.exit(0 if miss == 0 else 1)
