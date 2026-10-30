# Detailed finger-surface audit

Exact point-to-triangle Euclidean distance; optional generalized winding on closed oriented detailed mesh

Distances do not modify modeled validity and do not establish physical contact material or pickup/assembly success

| Saved subset | Poses | Contact points | Distance range(mm) | Within1um: points / both-contact poses |
|---|---:|---:|---:|---:|
|All modeled-valid|189|378|0.026450–1.326623|0 / 0|
|Selected|28|56|0.038037–1.326623|0 / 0|
|Unique CAD valid fits|137|274|0.026450–1.326623|0 / 0|

| Diagnostic threshold | All valid: points / both-contact poses | Selected: points / both-contact poses |
|---|---:|---:|
|0.001mm|0 / 0|0 / 0|
|0.1mm|154 / 43|33 / 13|
|0.25mm|240 / 92|42 / 19|

Detailed finger_vis is visual-only in the supplied XML. Collision uses the coarser finger mesh and separate pad boxes. Disagreement exposes a model-consistency issue; the detailed visual asset is not physical calibration.

1um is the existing intended-contact tolerance;0.1mm and0.25mm are diagnostic reporting thresholds, never tuned acceptance limits
