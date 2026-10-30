# NIST Task Board CAD provenance

The twenty loose-part meshes referenced by [cad_library.json](cad_library.json)
come from the NIST Task Board 1 STL archive:
[taskboard1stlzip](https://www.nist.gov/document/taskboard1stlzip).

Their existing conversion is implemented in
[`simulation/nist_task_board_1.py`](../simulation/nist_task_board_1.py):
millimetres to metres, centered mesh coordinates, declared axis conversion and
mesh simplification where required by the simulator. No fitted object scaling
or new mesh conversion was performed for publication.

The original local archive hash, conversion revision and dimensions are in
[`provenance.json`](../outputs/simulation/nist_task_board_1/assets/provenance.json).
The GitHub snapshot includes the twenty converted catalog meshes and the
explicit receiver/plate references needed for the October 29 grasp checks.
The [receiver asset manifest](../results/grasp/controlled_20261029/receiver_assets_manifest.json)
lists the nine required receiver meshes, the original virtual plate used to
derive its nominal finished holes, and that derivation receipt, with exact hashes.
These files remain at their existing asset paths; publication does not reconvert
or modify them. The finished plate is a derived simulation reference with the
recorded hole-size assumptions, rather than new physical mating validation.
The full archive, other board/fixture assets, captures and renderer dependencies remain local.
Refer to the upstream source for its terms; this notice does not assign a new
license to third-party assets.
