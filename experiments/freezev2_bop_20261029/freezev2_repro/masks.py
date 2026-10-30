"""External segmentation inputs for the isolated FreeZeV2 localization study.

This module never runs segmentation and never reads ground-truth poses. The
base candidate pool is the N+1 highest confidence detections *per segmentor*.
Segmentation confidence is not a replacement for the final pose score.
"""
from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


class MaskInputError(ValueError):
    """A malformed or incorrectly declared external mask input."""


class PredictedMasksUnavailable(FileNotFoundError):
    """A requested prediction input is absent; inference must report unavailable."""


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _integer(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MaskInputError(f"{name} must be an integer")
    if int(value) < minimum:
        raise MaskInputError(f"{name} must be >= {minimum}")
    return int(value)


def _compressed_counts(value: str | bytes) -> list[int]:
    """Decode the public COCO signed/delta-coded 5-bit count representation.

    The format is interoperable with pycocotools, which is optional here. Tests
    compare this decoder with that package when it is installed.
    """
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise MaskInputError("COCO RLE counts must be ASCII") from exc
    counts: list[int] = []
    index = 0
    while index < len(value):
        number, shift = 0, 0
        while True:
            if index >= len(value):
                raise MaskInputError("Truncated compressed COCO RLE")
            char = ord(value[index]) - 48
            index += 1
            if not 0 <= char <= 63:
                raise MaskInputError("Invalid compressed COCO RLE character")
            number |= (char & 31) << shift
            shift += 5
            if shift > 65:
                raise MaskInputError("Unreasonably large compressed COCO count")
            if not char & 32:
                if char & 16:
                    number |= -1 << shift
                break
        if len(counts) > 2:
            number += counts[-2]
        if number < 0:
            raise MaskInputError("COCO RLE count became negative")
        counts.append(number)
    return counts


def decode_rle(rle: Mapping) -> np.ndarray:
    """Return a HxW bool mask; COCO traverses pixels in column-major order."""
    if not isinstance(rle, Mapping) or "size" not in rle or "counts" not in rle:
        raise MaskInputError("Expected COCO RLE with size and counts")
    size = rle["size"]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise MaskInputError("RLE size must be [height, width]")
    height, width = (_integer(x, "RLE dimension", 1) for x in size)
    counts = rle["counts"]
    if isinstance(counts, (str, bytes)):
        counts = _compressed_counts(counts)
    elif isinstance(counts, (list, tuple)):
        counts = [_integer(x, "RLE count") for x in counts]
    else:
        raise MaskInputError("RLE counts must be integer runs or compressed text")
    if sum(counts) != height * width:
        raise MaskInputError("RLE run lengths do not cover the declared image")
    pixels = np.zeros(height * width, dtype=bool)
    offset = 0
    for run, length in enumerate(counts):
        if run % 2:
            pixels[offset:offset + length] = True
        offset += length
    return pixels.reshape((height, width), order="F")


def encode_rle(mask: np.ndarray) -> dict:
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise MaskInputError("Mask must be a binary 2D array")
    flat = mask.astype(bool).ravel(order="F")
    counts, last, length = [], False, 0
    for value in flat:
        if value == last:
            length += 1
        else:
            counts.append(length)
            length, last = 1, bool(value)
    counts.append(length)
    return {"size": list(mask.shape), "counts": counts}


@dataclass(frozen=True)
class LocalizationTarget:
    scene_id: int
    image_id: int
    object_id: int
    instance_count: int


@dataclass(frozen=True)
class MaskCandidate:
    scene_id: int
    image_id: int
    object_id: int
    score: float
    mask: np.ndarray
    segmentor: str
    source_id: str
    record_index: int
    source_kind: str = "predicted"
    source_file_sha256: str | None = None

    def __post_init__(self):
        for name in ("scene_id", "image_id", "record_index"):
            _integer(getattr(self, name), name)
        _integer(self.object_id, "object_id", 1)
        if not math.isfinite(float(self.score)):
            raise MaskInputError("Segmentation confidence must be finite")
        if self.source_kind not in {"predicted", "oracle"}:
            raise MaskInputError("Mask source_kind must be predicted or oracle")
        if not self.segmentor.strip() or not self.source_id.strip():
            raise MaskInputError("Mask segmentor and source_id provenance are mandatory")
        binary = np.asarray(self.mask)
        if binary.ndim != 2 or not np.isin(binary, [0, 1]).all():
            raise MaskInputError("Candidate mask must be binary and 2D")
        binary = binary.astype(bool, copy=True)
        binary.flags.writeable = False
        object.__setattr__(self, "mask", binary)

    @property
    def candidate_id(self) -> str:
        return f"{self.segmentor}:{self.scene_id:06d}:{self.image_id:06d}:{self.record_index}"

    def provenance(self) -> dict:
        return dict(candidate_id=self.candidate_id, segmentor=self.segmentor,
                    source_id=self.source_id, source_kind=self.source_kind,
                    source_file_sha256=self.source_file_sha256,
                    segmentation_score=float(self.score), record_index=self.record_index)


def load_localization_targets(path: str | Path) -> list[LocalizationTarget]:
    """Read official localization IDs/counts. Image-only target lists are invalid."""
    rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise MaskInputError("Localization target file must contain a nonempty list")
    targets, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or not {"scene_id", "im_id", "obj_id", "inst_count"} <= row.keys():
            raise MaskInputError("Official localization targets require scene_id, im_id, obj_id and inst_count; image-only lists are insufficient")
        target = LocalizationTarget(_integer(row["scene_id"], "scene_id"),
                                    _integer(row["im_id"], "im_id"),
                                    _integer(row["obj_id"], "obj_id", 1),
                                    _integer(row["inst_count"], "inst_count", 1))
        key = (target.scene_id, target.image_id, target.object_id)
        if key in seen:
            raise MaskInputError(f"Duplicate localization target: {key}")
        seen.add(key)
        targets.append(target)
    return sorted(targets, key=lambda t: (t.scene_id, t.image_id, t.object_id))


def load_mask_candidates(path: str | Path, *, segmentor: str, source_id: str,
                         source_kind: str = "predicted", allow_oracle: bool = False,
                         scene_id: int | None = None, image_id: int | None = None,
                         object_id: int | None = None,
                         target_keys: set[tuple[int, int, int]] | None = None) -> list[MaskCandidate]:
    """Load extended-COCO/BOP detection RLEs with explicit mask provenance.

    Filter image/object IDs before decoding masks when using a dataset-wide
    public prediction file. This prevents materializing all dense T-LESS masks
    merely to process one fixed image.
    """
    path = Path(path)
    if not path.is_file():
        raise PredictedMasksUnavailable(f"External predicted masks unavailable: {path}; no segmentation or GT fallback was performed")
    gt_path = any(part.lower() in {"mask_visib", "full_eval_gt", "eval_gt"} for part in path.parts)
    if source_kind not in {"predicted", "oracle"}:
        raise MaskInputError("Declare source_kind as predicted or oracle")
    if gt_path and source_kind != "oracle":
        raise MaskInputError("A ground-truth path cannot be declared as predicted masks")
    if source_kind == "oracle" and not allow_oracle:
        raise MaskInputError("Oracle masks require explicit allow_oracle=True and a separate diagnostic run")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig") as stream:
        rows = json.load(stream)
    if isinstance(rows, dict):
        rows = rows.get("detections", rows.get("predictions"))
    if not isinstance(rows, list):
        raise MaskInputError("Mask JSON must be a detection list or detections/predictions wrapper")
    checksum, candidates = file_sha256(path), []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MaskInputError("Every external mask detection must be an object")
        if any(k in row for k in ("gt_id", "GT_annotation_id", "visib_fract", "cam_R_m2c", "cam_t_m2c")) and source_kind != "oracle":
            raise MaskInputError("Ground-truth annotation fields found in predicted-mask input")
        if "image_id" in row and "im_id" in row and row["image_id"] != row["im_id"]:
            raise MaskInputError("Conflicting image_id and im_id")
        image = row.get("image_id", row.get("im_id"))
        required = {"scene_id", "category_id", "score", "segmentation"}
        if not required <= row.keys() or image is None:
            raise MaskInputError("Detections need scene_id, image_id/im_id, category_id, score and segmentation")
        row_scene = _integer(row["scene_id"], "scene_id")
        row_image = _integer(image, "image_id")
        row_object = _integer(row["category_id"], "category_id", 1)
        if ((scene_id is not None and row_scene != scene_id)
                or (image_id is not None and row_image != image_id)
                or (object_id is not None and row_object != object_id)
                or (target_keys is not None and (row_scene, row_image, row_object) not in target_keys)):
            continue
        record_kind = row.get("source_kind", source_kind)
        if record_kind != source_kind:
            raise MaskInputError("Detection source_kind disagrees with declared file provenance")
        candidates.append(MaskCandidate(row_scene, row_image, row_object,
                                        float(row["score"]), decode_rle(row["segmentation"]),
                                        segmentor, source_id, index, source_kind, checksum))
    return candidates


def select_localization_candidates(candidates: Iterable[MaskCandidate], *, scene_id: int,
                                   image_id: int, object_id: int, instance_count: int,
                                   allow_oracle: bool = False) -> list[MaskCandidate]:
    """The base FreeZeV2 pool: N+1 confidence-ranked candidates per segmentor.

    No extra mask NMS/deduplication, confidence fusion, or test-GT selection is
    inserted here. Final pose ranking is performed after rigid registration.
    """
    count = _integer(instance_count, "instance_count", 1)
    groups: dict[str, list[MaskCandidate]] = {}
    for candidate in candidates:
        if (candidate.scene_id, candidate.image_id, candidate.object_id) != (scene_id, image_id, object_id):
            continue
        if candidate.source_kind == "oracle" and not allow_oracle:
            raise MaskInputError("Oracle masks cannot enter a predicted-mask benchmark")
        if candidate.mask.any():
            groups.setdefault(candidate.segmentor, []).append(candidate)
    result = []
    for segmentor in sorted(groups):
        ordered = sorted(groups[segmentor], key=lambda c: (-c.score, c.record_index, c.candidate_id))
        result.extend(ordered[:count + 1])
    return result

