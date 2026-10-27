"""Local Ollama bridge. Windows owns its loopback API; no external API calls."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
MODEL = "qwen3.5:9b"
EXPECTED_DIGEST = "6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7"
QUEUE = HERE.parents[1] / "outputs/grconvnet_vlm_pickup_20261024/bridge_queue"
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "num_predict": 512,
           "repeat_penalty": 1., "presence_penalty": 0., "frequency_penalty": 0.,
           "top_k": 0, "top_p": 1., "min_p": 0.}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def winpath(path):
    text = str(Path(path).resolve())
    if text.startswith("/mnt/d/"):
        return "D:/"+text[len("/mnt/d/"):]
    if text.startswith("/mnt/c/"):
        return "C:/"+text[len("/mnt/c/"):]
    return text


def api(route, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    request = Request("http://127.0.0.1:11434"+route, body, {"Content-Type": "application/json"})
    with urlopen(request, timeout=900) as response:
        return json.load(response)


def windows_main(request_path, response_path):
    spec = json.loads(Path(request_path).read_text())
    tag = next(m for m in api("/api/tags")["models"] if m["name"] == MODEL)
    if tag["digest"] != EXPECTED_DIGEST:
        raise ValueError("Local VLM checkpoint changed")
    if spec.get("action") == "inventory":
        result = {"model": MODEL, "digest": tag["digest"], "runtime": api("/api/version"), "options": OPTIONS}
    else:
        image = Path(spec["image_file"])
        if digest(image) != spec["image_sha256"]:
            raise ValueError("Changed visual input")
        wire = {"model": MODEL, "stream": False, "think": False, "keep_alive": "30m",
                "options": OPTIONS, "format": {"type": "object", "properties": {
                    "ranked_ids": {"type": "array", "items": {"type": "integer", "enum": spec["allowed_ids"]},
                                   "minItems": len(spec["allowed_ids"]), "maxItems": len(spec["allowed_ids"])},
                    "observations": {"type": "array", "items": {"type": "object", "properties": {
                        "id": {"type": "integer", "enum": spec["allowed_ids"]}, "observation": {"type": "string"}},
                        "required": ["id", "observation"]}}}, "required": ["ranked_ids", "observations"]},
                "messages": [{"role": "user", "content": spec["prompt"],
                              "images": [base64.b64encode(image.read_bytes()).decode()]}]}
        started = time.perf_counter()
        raw = api("/api/chat", wire)
        result = {"raw_response": raw, "api_wall_seconds": time.perf_counter()-started,
                  "wire_sha256": hashlib.sha256(json.dumps(wire).encode()).hexdigest(),
                  "model_digest": tag["digest"], "image_sha256": spec["image_sha256"]}
    Path(response_path).write_text(json.dumps(result, indent=2)+"\n")


def bridge(spec, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request, response = folder / "request.json", folder / "response.json"
    request.write_text(json.dumps(spec, indent=2)+"\n")
    started = time.perf_counter()
    QUEUE.mkdir(parents=True,exist_ok=True)
    job=QUEUE/(uuid.uuid4().hex+".job.json")
    temporary=job.with_suffix(".tmp")
    temporary.write_text(json.dumps({"request":winpath(request),"response":winpath(response)}))
    temporary.rename(job)
    while not response.exists():
        if time.perf_counter()-started>930:
            raise TimeoutError("Local Windows VLM worker did not respond")
        time.sleep(.1)
    result = json.loads(response.read_text())
    if "bridge_error" in result:
        raise RuntimeError(result["bridge_error"])
    result["bridge_wall_seconds"] = time.perf_counter()-started
    return result


def rank(sheet, ids, folder):
    prompt = (HERE / "grasp_rank_prompt.txt").read_text().strip()+"\nSupplied IDs: "+json.dumps(ids)
    response = bridge({"image_file": winpath(sheet), "image_sha256": digest(sheet),
                       "allowed_ids": ids, "prompt": prompt, "prompt_file_sha256": digest(HERE / "grasp_rank_prompt.txt")}, folder)
    raw = response["raw_response"]
    text = raw.get("message", {}).get("content", "")
    response["raw_text"] = text
    response["status"] = "invalid_output"
    try:
        parsed = json.loads(text)
        order = parsed["ranked_ids"]
        observations=parsed["observations"]
        if (not raw.get("done") or raw.get("done_reason") != "stop" or raw.get("message", {}).get("thinking")
                or len(order) != len(ids) or any(type(x) is not int for x in order) or sorted(order) != sorted(ids)):
            raise ValueError("Ranking must finish and contain each supplied integer ID once")
        if (len(observations)!=len(ids) or sorted(item["id"] for item in observations)!=sorted(ids)
                or any(not isinstance(item["observation"],str) or not item["observation"].strip() for item in observations)):
            raise ValueError("Need one nonempty observation per supplied ID")
        response.update(status="ok", ranked_ids=order, parsed=parsed)
    except (ValueError, KeyError, TypeError) as error:
        response["parse_error"] = str(error)
    (Path(folder) / "ranking.json").write_text(json.dumps(response, indent=2)+"\n")
    return response


if __name__ == "__main__":
    if sys.argv[1]=="--worker":
        QUEUE.mkdir(parents=True,exist_ok=True)
        while not (QUEUE/"stop").exists():
            for job in sorted(QUEUE.glob("*.job.json")):
                try:
                    item=json.loads(job.read_text())
                    try:
                        windows_main(item["request"],item["response"])
                    except Exception as error:
                        Path(item["response"]).write_text(json.dumps({"bridge_error":str(error)}))
                    job.rename(job.with_suffix(".done"))
                except Exception as error:
                    print(str(error),flush=True)
            time.sleep(.1)
    else:
        windows_main(sys.argv[1], sys.argv[2])
