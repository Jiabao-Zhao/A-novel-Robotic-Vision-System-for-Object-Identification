# Current NIST CAD/Depth Pipeline

1. The target request retrieves a CAD model from `CAD/cad_library.json` by cosine
   similarity between supplied text embeddings.
2. Depth localization separates observed objects. Calibrated projection provides
   the RGB crops and corresponding camera-frame partial point clouds.
3. DINO global/foreground-patch appearance and native-scale CAD/depth agreement
   identify the matching candidate and estimate the aligned CAD pose.
4. Grasp research generates and checks candidates using that aligned CAD and the
   partial observation. Physical holding and assembly quality remain separate
   questions under investigation.

NIST Task Board 1 is the only enabled scene/CAD-generation family. Individual
studies declare their chosen NIST products, sizes, scene variants and cameras.
Keep the shared simulator/camera/Panda backend and required installed assets.

The older likelihood classifier and CAD-description/SAM3 scene-selection protocol
are not the active pipeline. Optional saved baseline comparisons do not replace
the depth-localization stage or establish an end-to-end deployed system.

Use metres and explicit camera/CAD/world/robot transforms. Preserve native scale,
unavailable fits, tied candidates and symmetry/visibility limits. Ground-truth
identities, masks and simulator poses are evaluation-only. Report the searched
pose degrees of freedom, not an unsupported full-6D claim.

The cleanup changes source organization and asset paths, not saved numerical
results. The cleanup record documents removed artifacts and preserved shared
inputs. Historical manifests/source locks remain descriptions of their original
runs, not certification of the reorganized checkout.
