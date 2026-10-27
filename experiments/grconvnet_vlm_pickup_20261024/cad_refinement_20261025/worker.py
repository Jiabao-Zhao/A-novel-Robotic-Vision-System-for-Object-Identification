"""New local queue for the frozen Windows Ollama client; no external API."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.grconvnet_vlm_pickup_20261024.vlm_client import windows_main

QUEUE = ROOT / "outputs/grconvnet_vlm_pickup_20261024/cad_refinement_20261025/bridge_queue"
QUEUE.mkdir(parents=True, exist_ok=True)
while not (QUEUE/"stop").exists():
    for job in sorted(QUEUE.glob("*.job.json")):
        item = json.loads(job.read_text())
        try:
            windows_main(item["request"], item["response"])
        except Exception as error:
            Path(item["response"]).write_text(json.dumps({"bridge_error": str(error)}))
        job.rename(job.with_suffix(".done"))
    time.sleep(.1)
