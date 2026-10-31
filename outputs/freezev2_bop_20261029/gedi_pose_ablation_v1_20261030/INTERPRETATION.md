# Measured interpretation

The finite-row GeDi repair recovered valid-format pose outputs for all four targets that had no pose in the strict baseline. Both repaired arms saved seven poses from the same eleven candidate masks. This is numerical recovery, not seven correct localizations.

On this fixed failure-heavy cohort, official pose AR was0.4762% for the saved baseline,14.2381% for repair224, and5.1905% for repair476. All seven requested instances remained in every official score denominator. These are BOP19 pose recalls, not segmentation AP or CAD-identification accuracy, and not the complete T-LESS benchmark.

The strongest recovery was CAD30 at scene1/image1: repair224 reached90.0% per-target AR (VSD80%, MSSD100%, MSPD90%). CAD476 fell to0% for that same mask25. The two saved refined poses differ by129.72degrees and45.37mm; this delta uses saved predictions only. A higher-resolution visual descriptor changed correspondences and the recovered pose; it did not guarantee better registration. No thresholds or settings were changed in response.

CAD476 improved CAD28 scene4/image0 from0 to16.67% per-target AR and changed selection from background mask7526 to foreground mask7525, but its MSSD and MSPD recalls were still0. The gain was partial VSD agreement, not a fully accurate6D pose. The CAD30 loss outweighed such gains. CAD27 remained0AR in both repaired arms despite successful feature extraction and pose output.

The recorded observed GeDi failure at scene3/image476/CAD8 gained6.33% per-target AR with repair224 and3.67% with repair476; MSSD/MSPD remained0. Its unchanged selected mask covers several objects. The scene3/image5/CAD8 control selected background/fiducial masks and remained0AR. The visual review records these unresolved input/ranking problems.

The two computationally healthy CAD1 controls reproduced their original features, scores and poses exactly in repair224. Here healthy means numerically executable, not GT-correct. The78 focused tests passed; all saved CSV poses were reconciled with ledgers and all360 threshold denominators/matches were checked against official scores.

These results establish a working numerical repair and separate the476 contribution. They do not establish a full-dataset gain or reproduce the public repository's performance. Original all-seven baseline outputs remain unchanged. No full-suite rerun, parameter tuning, model training, dataset/model download, environment installation, commit or remote push was performed.
