# LIBERO CAD asset provenance

The LIBERO CAD catalog does not redistribute simulator meshes. At runtime,
`simulation.libero_cad` resolves each installed asset from the current LeRobot /
LIBERO cache beneath `~/.cache/libero/assets`.

All ten LIBERO-Object target meshes listed below come from the
[NVIDIA HOPE dataset](https://github.com/swtyree/hope-dataset), distributed by
the Hugging Face dataset repository `lerobot/libero-assets`. They were verified
against asset revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` and retain
the source [CC BY-NC-SA 4.0 license](https://creativecommons.org/licenses/by-nc-sa/4.0/).

| Target | Installed asset relative to `~/.cache/libero/assets` | Scale to m | CAD up axis | Verified OBJ SHA-256 |
|---|---|---:|:---:|---|
| Alphabet soup | `stable_hope_objects/alphabet_soup/textured.obj` | `0.01` | Z | `9f83394a8b6d243b2133be244b3de83e9318401e53a5768063031795bc6919a0` |
| Cream cheese | `stable_hope_objects/cream_cheese/cream_cheese.obj` | `0.008` | Z | `79ca603fd960a95643cf7532b702343c1b389af5cd86b84dfeb931a2cc23a4b3` |
| Salad dressing | `stable_hope_objects/salad_dressing/textured.obj` | `0.01` | Y | `0045f55b86068e26888bcff4ea8a0eaebe2f3ec791349cae0d593066d68b9ba7` |
| BBQ sauce | `stable_hope_objects/bbq_sauce/bbq_sauce.obj` | `0.0077` | Y | `69030e437ba6d8970e925f1b5344b80d2a67511658c2625dfbeb508f53e34417` |
| Ketchup | `stable_hope_objects/ketchup/textured.obj` | `0.01` | Y | `dd07788b2fc0ece118cd97d544cea0fa6e87f56440f69d9bc13b876143490414` |
| Tomato sauce | `stable_hope_objects/tomato_sauce/textured.obj` | `0.01` | Z | `b75cd4063af13da2c3c95f4ed6cc2fbdc00f51730b49b704feb6f1ef362204bf` |
| Butter | `stable_hope_objects/butter/butter.obj` | `0.0075` | Z | `4d32a23384f059ee79cc6c9d4d18d2d12c171c13bd89aaa4829a332123a1bd4a` |
| Milk | `stable_hope_objects/milk/textured.obj` | `0.0075` | Y | `caff7624f6aa1183166344350f3491587c53ed42ca7a57058b6e52796cf42e1f` |
| Chocolate pudding | `stable_hope_objects/chocolate_pudding/textured.obj` | `0.01` | Z | `08ea9b9106f9595db3b8544f0c1c4b55e3b62303191f2cf8a75aa677abc8b237` |
| Orange juice | `stable_hope_objects/orange_juice/textured.obj` | `0.0075` | Y | `8f2296913bd1a82160fa1291ddb72f4eeb3703437513481c8b840d1933c460f1` |

These OBJ files have the same unscaled vertex bounds as the compiled MuJoCo
visual meshes referenced by the installed object XML files. LIBERO's declared
per-object scale is applied when loading a CAD prior.

Because these are the same meshes used to create the rendered simulator
objects, results must be labeled as an **exact simulator CAD prior** condition.
The pipeline selects a prior from the VLM semantic classification and never
reads simulator object identity, instance transform, segmentation, or
ground-truth pose.
