"""Sealed disk index of external RLEs; decode only requested N+1 masks.

The database is committed/closed before its byte hash is sealed, then reopened
read-only. The sidecar is an integrity check, not a worker-liveness signal.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import sqlite3

from .data import sha256
from .dataset_checks import iter_json_array
from .durable import atomic_json, digest, read_json
from .masks import MaskCandidate, decode_rle


class MaskIndex:
    def __init__(self, path, sources):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar = self.path.with_name(self.path.name + '.integrity.json')
        self.sources = sources
        self.identity = [{key: source[key] for key in ('path', 'segmentor', 'source_id', 'sha256')} for source in sources]
        self.identity_digest = digest(self.identity)
        self.db = None
        if not sources:
            raise ValueError('At least one explicit predicted-mask source is required')
        for source in sources:
            if sha256(source['path']) != source['sha256']:
                raise ValueError('Predicted-mask source checksum changed')
            if source.get('source_kind', 'predicted') != 'predicted':
                raise ValueError('Suite accepts predicted masks only; oracle diagnostic is separate')
        wal = self.path.with_name(self.path.name + '-wal')
        if wal.exists() and wal.stat().st_size:
            raise ValueError('Mask-index has an external WAL; reconcile its writer before reuse')
        if self.sidecar.exists():
            self.seal = read_json(self.sidecar)
            if self.seal.get('schema') != 'sealed-predicted-mask-index-v3' or self.seal.get('source_identity') != self.identity_digest:
                raise ValueError('Mask-index source identity differs')
            if not self.path.is_file() or sha256(self.path) != self.seal.get('database_sha256'):
                raise ValueError('Mask-index database checksum changed; cached RLEs are not trusted')
        else:
            self._build_or_recover()
            self.seal = read_json(self.sidecar)
        try:
            self.db = self._read_only()
            identity = self.db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            complete = self.db.execute("SELECT value FROM meta WHERE key='complete'").fetchone()
            count = self.db.execute('SELECT COUNT(*) FROM masks').fetchone()[0]
            if identity != (self.identity_digest,) or complete != ('1',) or count != self.seal['rows']:
                raise ValueError('Mask-index sealed metadata/row count differs')
            self._fingerprint = self._stat()
        except BaseException:
            self.close()
            raise

    def _read_only(self):
        return sqlite3.connect(self.path.as_uri() + '?mode=ro', uri=True)

    def _stat(self):
        stat = self.path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def _source_rows(self):
        """Canonical rows in source/record order, with bounded parser memory."""
        for source_index, source in enumerate(self.sources):
            for record, row in enumerate(iter_json_array(source['path'])):
                if not isinstance(row, dict):
                    raise ValueError('Public mask detections must be objects in a JSON array')
                if any(key in row for key in ('gt_id', 'GT_annotation_id', 'visib_fract', 'cam_R_m2c', 'cam_t_m2c')):
                    raise ValueError('GT annotation fields in predicted-mask source')
                if 'image_id' in row and 'im_id' in row and row['image_id'] != row['im_id']:
                    raise ValueError('Conflicting external image_id/im_id')
                scene, image, object_id = row['scene_id'], row.get('image_id', row.get('im_id')), row['category_id']
                if (not all(isinstance(value, int) and not isinstance(value, bool) for value in (scene, image, object_id))
                        or scene < 0 or image < 0 or object_id < 1):
                    raise ValueError('Invalid external scene/image/object IDs')
                score = float(row['score'])
                if not math.isfinite(score):
                    raise ValueError('Nonfinite external confidence')
                yield (source_index, scene, image, object_id, score, record,
                       json.dumps(row['segmentation'], separators=(',', ':')))

    def _write_seal(self, count, *, recovered):
        # Construction's SQLite handles have closed before this byte checksum.
        atomic_json(self.sidecar, {'schema': 'sealed-predicted-mask-index-v3',
                                  'source_identity': self.identity_digest,
                                  'database_sha256': sha256(self.path), 'rows': count,
                                  'recovered_after_row_verification': recovered})

    def _build_or_recover(self):
        connection = sqlite3.connect(self.path)
        recovered, count = False, 0
        try:
            connection.execute('PRAGMA journal_mode=DELETE')
            connection.execute('PRAGMA synchronous=FULL')
            connection.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
            previous = connection.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
            complete = connection.execute("SELECT value FROM meta WHERE key='complete'").fetchone()
            if previous and previous != (self.identity_digest,):
                raise ValueError('Mask-index source identity differs')
            if complete == ('1',):
                # Recover commit-before-sidecar interruptions or old v2 caches
                # only after proving every SQL row against unchanged sources.
                stored = connection.execute('SELECT source,scene,image,object,score,record,rle FROM masks ORDER BY source,record')
                for expected in self._source_rows():
                    if stored.fetchone() != expected:
                        raise ValueError('Unsealed mask-index rows differ from the public source')
                    count += 1
                if stored.fetchone() is not None:
                    raise ValueError('Unsealed mask-index has extra rows')
                recovered = True
            else:
                connection.execute('BEGIN IMMEDIATE')
                connection.execute('DROP TABLE IF EXISTS masks')
                connection.execute('CREATE TABLE masks(source INTEGER,scene INTEGER,image INTEGER,object INTEGER,score REAL,record INTEGER,rle TEXT)')
                for row in self._source_rows():
                    connection.execute('INSERT INTO masks VALUES(?,?,?,?,?,?,?)', row)
                    count += 1
                connection.execute('CREATE INDEX target_lookup ON masks(scene,image,object,source,score DESC,record)')
                connection.execute("INSERT OR REPLACE INTO meta VALUES('identity',?)", (self.identity_digest,))
                connection.execute("INSERT OR REPLACE INTO meta VALUES('complete','1')")
                connection.commit()
            if connection.execute('PRAGMA quick_check').fetchone() != ('ok',):
                raise ValueError('SQLite integrity preflight failed for the mask index')
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()
        self._write_seal(count, recovered=recovered)

    def _check_unchanged(self):
        if self.db is None:
            raise RuntimeError('Mask index is closed')
        if self._stat() != self._fingerprint:
            if sha256(self.path) != self.seal['database_sha256']:
                raise ValueError('Mask-index database checksum changed before query')
            # A same-byte replacement needs a fresh read-only SQLite handle.
            self.db.close()
            self.db = self._read_only()
            self._fingerprint = self._stat()

    def candidates(self, scene_id, image_id, object_id, instance_count):
        if isinstance(instance_count, bool) or not isinstance(instance_count, int) or instance_count < 1:
            raise ValueError('Official instance_count must be a positive integer')
        self._check_unchanged()
        groups = {}
        for source_index, source in enumerate(self.sources):
            groups.setdefault(source['segmentor'], [])
            rows = self.db.execute('SELECT score,record,rle FROM masks WHERE scene=? AND image=? AND object=? AND source=? ORDER BY score DESC,record',
                                   (scene_id, image_id, object_id, source_index))
            accepted = 0
            for score, record, rle in rows:
                mask = decode_rle(json.loads(rle))
                if not mask.any():
                    continue
                groups[source['segmentor']].append(MaskCandidate(scene_id, image_id, object_id, score, mask,
                    source['segmentor'], source['source_id'], record, 'predicted', source['sha256']))
                accepted += 1
                if accepted >= instance_count + 1:
                    break
        result = []
        for segmentor in sorted(groups):
            result.extend(sorted(groups[segmentor], key=lambda value: (-value.score, value.record_index, value.candidate_id))[:instance_count + 1])
        return result

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None
