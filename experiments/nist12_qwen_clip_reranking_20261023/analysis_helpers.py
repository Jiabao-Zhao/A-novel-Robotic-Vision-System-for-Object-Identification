"""JSON/CSV output and candidate ranking shared by descriptor experiments."""
import csv
import hashlib
import json
import math
from pathlib import Path

ATOL = 1e-12


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(folder, stem, rows):
    folder = Path(folder)
    save(folder / (stem + ".json"), rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (folder / (stem + ".csv")).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)


def rank_candidates(candidates, expected_ids=None):
    """Rank without an identity, score cutoff, or arbitrary tie break."""
    by_id = {int(row["proposal_id"]): row for row in candidates}
    assert len(by_id) == len(candidates), "Duplicate candidates cannot be ranked."
    if expected_ids is not None:
        assert set(by_id) == set(expected_ids), "All competitors must be represented."
    assert by_id
    if any(row["score"] is None for row in candidates):
        return dict(available=False, top_proposal_ids=[], top_score=None, gap=None,
                    winner_proposal_id=None, ordered_proposal_ids=[], nonpositive_maximum=None)
    ordered = sorted(candidates, key=lambda row: (-row["score"], int(row["proposal_id"])))
    top = ordered[0]["score"]
    top_ids = [int(row["proposal_id"]) for row in ordered
               if math.isclose(top, row["score"], rel_tol=0, abs_tol=ATOL)]
    return dict(available=True, top_proposal_ids=top_ids, top_score=top,
                winner_proposal_id=top_ids[0] if len(top_ids) == 1 else None,
                gap=top - ordered[1]["score"] if len(ordered) > 1 else None,
                ordered_proposal_ids=[int(row["proposal_id"]) for row in ordered],
                nonpositive_maximum=top <= 0)


def outcome(rank, true_id):
    if not rank["available"]:
        return "unavailable"
    if len(rank["top_proposal_ids"]) > 1:
        return "tie"
    return "correct" if rank["winner_proposal_id"] == true_id else "wrong"
