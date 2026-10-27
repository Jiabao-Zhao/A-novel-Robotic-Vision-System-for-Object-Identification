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
The GitHub snapshot includes only the twenty converted catalog meshes. The full
archive, board/fixture bank, captures and renderer dependencies remain local.
Refer to the upstream source for its terms; this notice does not assign a new
license to third-party assets.
