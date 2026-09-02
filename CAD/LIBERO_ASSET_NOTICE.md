# LIBERO CAD asset provenance

The LIBERO CAD catalog does not redistribute simulator meshes. At runtime,
`simulation.libero_cad` resolves the installed asset from the current LeRobot /
LIBERO cache beneath `~/.cache/libero/assets`.

The milk experiment uses:

- asset: `stable_hope_objects/milk/textured.obj`
- Hugging Face dataset repository: `lerobot/libero-assets`
- tested asset revision: `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6`
- verified OBJ SHA-256: `caff7624f6aa1183166344350f3491587c53ed42ca7a57058b6e52796cf42e1f`
- source dataset: [NVIDIA HOPE](https://github.com/swtyree/hope-dataset)
- source license: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- LIBERO-applied scale: `0.0075` source units to metres

This is the exact geometry used to create LIBERO's rendered milk mesh. Results
must therefore be labeled as an **exact simulator CAD prior** condition. The
pipeline selects this prior from the VLM semantic classification and never
reads the simulator object's identity, instance transform, segmentation, or
ground-truth pose.
