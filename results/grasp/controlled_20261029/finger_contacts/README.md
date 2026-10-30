# Preserved modeled finger contacts: 37/96

This is saved geometric evidence: **37/96 placements** retain a selected grasp, with 75 valid modeled candidates from 9600 proposals. These counts do not measure pickup or assembly success. VLM selection is excluded; pickup and assembly execution remain paused.

- [Selected-grasp gallery](gallery/index.html) and [report](report.md).
- [Compact per-placement decisions](decisions.csv), [saved summary](summary.json), and [byte-preservation manifest](publication_manifest.json).
- [Complete saved results](results.json.gz). Each decision row links its complete selection and full candidate diagnostics, stored as `.json.gz` files.

Every copied source file is byte-identical after decompression. The manifest records original paths, exact source and stored SHA256 values, byte counts, and the pre-copy tree inventory. The decisions table is derived directly from saved values and does not run or change selection. Copied historical reports retain their original text and local source links; use this index for included files. Gallery HTML and PNGs are copied together and require no compressed JSON to open locally.

Read compressed JSON in Python without rewriting the source:

```python
import gzip, json
with gzip.open('results.json.gz', 'rt', encoding='utf-8') as f:
    placements = json.load(f)
```

Run `python -m experiments.cad_grasp_controlled_20261029.publish verify <this-directory>` from the repository root to verify every stored file and its decompressed source hash. Source paths in historical evidence may name the original local machine; they are provenance, not required extraction paths. Native CAD meshes remain at their existing tracked repository locations. No weights, raw captures or installed dependencies are copied.
