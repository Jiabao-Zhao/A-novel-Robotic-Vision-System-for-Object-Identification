# Analysis from a GitHub or cloud checkout

Start with [README.md](../README.md),
[simulation/pipeline_protocol.md](../simulation/pipeline_protocol.md),
[the experiment index](../experiments/README.md), and
[the published evidence index](../results/README.md).

## Available for analysis

The repository contains the current framework source and selected exact copies
of saved result tables, protocols and images under `results/`. The evidence
manifest links every copied file to its original local path and SHA256. Use
the small summaries first, then read per-task/per-object CSVs for concrete
failures. The thirteen-megabyte grasp decision file is detailed evidence;
filter it to a scene or part before discussing individual hand poses.

The latest [October 28 grasp run](../outputs/cad_grasp_selection_20261028/runs/20261028T190844Z/README.md)
is also published at its original output paths. It contains all 96 case decisions
and 9,600 proposal diagnostics, including failed searches, plus the eight gallery
sheets and derived CAD validation meshes. Its publication manifest preserves
the saved evidence hashes. The gallery HTML can be opened after cloning; GitHub
displays the linked PNG sheets directly. Saved observation/calibration inputs
and installed dependencies outside that run are still required for a rerun.

The pipeline is target-CAD retrieval by cosine similarity, depth localization,
calibrated RGB crops, DINO/depth matching and CAD alignment, followed by grasp
research. Some tested fitting and fusion code lives in the historically named
`experiments/nist12_qwen_clip_reranking_20261023/` directory. Its directory name
does not make Qwen or CLIP mandatory in the current framework.

The current grasp pipeline makes the final selection itself: mandatory geometry
checks, then moment arm, surface centrality, downward preference and a
deterministic candidate-ID tie break. VLM ranking/selection is not a pipeline
stage. Historical VLM studies and captions do not define the current policy.
Generation, pickup and assembly remain paused during mating-goal review.
The [official NIST source audit](nist_task_board_1_reference.md) supersedes the
earlier horizontal waterproof reference and wrong USB/DSUB and M4/M12 placement
assignments. Historical results remain unchanged. Saved native CAD transforms
are available locally and require explicit conversion to our component frames.

## Questions the evidence can answer

1. Which pin sizes or connector geometries fail, and at which stage: proposal,
   shortlist retrieval, appearance matching, unavailable depth fit or geometry
   ranking? Compare fixed and introduced errors on the same saved requests.
2. When does adding DINO appearance to depth change a correct geometric match,
   and what do the illustrated single-view cases reveal about scale, visible
   cross-section, contact clearance or partial observation?
3. Which aligned-CAD placements lose every grasp under the one-degree contact
   setting, and which modeled rejection explains each loss? Compare this with
   the fifteen-degree reference without calling either condition a pickup test.
4. In the October 28 selector, which checks reject the complete poses before
   priority selection? Compare the 4/96 selected placements with earlier studies
   only after accounting for their different budgets and contact conventions.

## Interpretation requirements

- Cite the result file and exact scene/part/method when making a numerical claim.
- Keep supplied CAD identity, assumed correspondence and observed CAD pose
  distinct from a measured full retrieval-to-grasp result.
- Separate common-crop matching from native full-scene pipelines. Keep native
  failures, declared compatibility controls, ties, unavailability, absence and
  duplicate-instance ambiguity visible.
- Preserve metric scale and state fitted degrees of freedom. Single-view partial
  clouds and symmetric parts can leave orientation unresolved.
- Simulator labels and poses are audit evidence only. Do not use them to improve
  recognition, pose fitting, grasp proposals or thresholds.
- Original source hashes describe the original run. Source and asset paths were
  reorganized on October 27; old source locks do not validate the new checkout.
- The robustness scope receipt is not completed accuracy evaluation. Candidate
  grasp availability is not physical holding, pickup or assembly accuracy.

## Local data and execution limits

The twenty native-scale CAD catalog meshes are included for inspection. Large
captures, full scene/CAD-render banks, complete pose banks, model checkpoints and
installed simulator/baseline dependencies remain local. Some tests and runners
therefore require local assets that a cloud checkout does not contain. Report
missing dependencies rather than inventing inputs or claiming a reproduction.
The full interactive retrieval-to-grasp runtime has not been validated.

Exact-copied historical reports retain their original relative links. Some point
to local-only galleries; use the published evidence index for included images.

The old C: experiment workspace was removed. Required local inputs now live in
`outputs/datasets/nist_task_board/legacy_workspace_inputs/` on D:. Existing path
readers map that exact retired prefix for immutable historical records. XML
path changes have a separate verified receipt; original full-run source locks
remain historical. See [the migration record](c_workspace_cleanup_20261027.json).

Do not restore retired likelihood classifiers or non-NIST layouts. Do not
download models, start paid APIs, alter environments or command a robot as a
side effect of reading or testing the repository. This handoff supplies evidence
for analysis; it does not request fresh experiments or robot execution.
