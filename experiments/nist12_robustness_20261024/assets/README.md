# Robustness asset inventory

This is a read-only inventory, not an experiment result. No model, robot or camera was called.

The four recommended new classification targets are BNC Female, USB Female, DSUB Female and Waterproof Female. Their native-meter meshes and source conversion receipts are already local. They have no exact `target` field matches in the searched experiment JSON, but were present as fixed fixtures in prior scene construction; describe them as newly queried CADs, not globally unseen project assets. Medium/small gears, M8/M12 nuts and RJ45 Male have prior classification history.

A cached real T-LESS subset contains 30 CADs and 69 RGB-D frames across 20 scene directories. Scene 1 has 50 frames; each remaining scene has one cached frame. Selecting the first available frame per scene yields a deterministic 20-scene pilot without selecting on labels or outcomes. This is not a full BOP evaluation. The per-frame camera JSON supplies intrinsics and depth scale; only those fields may reach inference. Camera extrinsics and object annotations are evaluation-only.

The existing physical camera capture has metric depth and intrinsics, but only one scene and no independent object/mask ground truth. Historical VLM/registration outputs disagree on object IDs and cannot define truth. It can support an exploratory pilot after explicit manual audit, not a robustness headline.

See `inventory.json` for paths, hashes, dimensions and the deterministic real-frame list.

## Rejection and ambiguity evaluation

Freeze decisions using separate development scenes. Raw scores are not probabilities and should not share one numerical cutoff across different methods. If no rejection calibration is ready, report score/threshold curves and current forced-choice false selections on target-absent queries without claiming a reliable rejection gate. Do not tune the cutoff on the held-out absent scenes.

For duplicate targets, any correct matching instance is a correct class localization; for a singular human request, multiple plausible distinct instances should trigger clarification. Evaluate both facts separately. Preserve candidate masks and scores so a wrong object, missed duplicate, over-segmentation fragment and unresolved request remain distinguishable.
