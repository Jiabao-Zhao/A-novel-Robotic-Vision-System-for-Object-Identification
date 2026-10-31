# Public source availability: FreeZe and FreeZeV2

Source audit verified on **2026-10-30 at 12:20 America/New_York** (16:20 UTC). This is a bounded public-source search, not proof that no private or unindexed implementation exists. This document records research findings; it does not claim that any inference or evaluation has completed in this experiment.

The task is an explicitly authorized isolated BOP reproduction. Root AGENTS.md, README.md, simulation/pipeline_protocol.md and experiments/README.md were read. No nested AGENTS.md was found for this experiment. Historical experiments, saved results and NIST/grasp production code remain separate.

## Author-controlled sources checked

- [FreeZe V1 paper](https://arxiv.org/abs/2312.00947) and its [ECCV version](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09634.pdf).
- [FreeZeV2 paper](https://arxiv.org/abs/2506.09784).
- [Author project website](https://andreacaraffa.github.io/freeze/) and its Code link to [andreacaraffa/freeze](https://github.com/andreacaraffa/freeze).
- Official repository main files, branches, tags, releases, issues, pull requests and all four public forks.
- Andrea Caraffa, Davide Boscaini, Fabio Poiesi and TeV-FBK public repository inventories, including related GeDi/dGeDi and pose projects.

The official `andreacaraffa/freeze` main revision observed was `c59c02f6c5c5447219d2ff2d5f4ef94759d1bbfc`, dated 2025-05-12. The public repository contains the project website: README, index.html and static CSS/JS/images/videos. No Python/C++/CUDA inference files, executable environment configuration or model-based pose runner was found. Only main was listed; no tags, releases or pull requests were listed.

The four public forks inspected were bugfreee/freeze, treeone246/freeze, Blessorzzz/freeze and Dart47-gift/freeze. Their inspected main trees matched the official website tree and contained no inference implementation.

## Direct author replies and release requests

| Evidence | Verified content |
|---|---|
| [Issue 1, author reply 2024-09-25](https://github.com/andreacaraffa/freeze/issues/1#issuecomment-2373973601) | The repository owner anticipated release in the coming weeks without specifying a date. This is a release intention, not released code. |
| [Issue 5, author reply 2025-09-25](https://github.com/andreacaraffa/freeze/issues/5#issuecomment-3334362280) | The owner offered to arrange a call about research support. The reply supplies no inference-source link or release date. |
| [Issue 4: FreeZe V2 open-source request](https://github.com/andreacaraffa/freeze/issues/4) | Open, no comments at inspection. |
| [Issue 7: fusion-code request](https://github.com/andreacaraffa/freeze/issues/7) | Open, no comments at inspection. |
| [Issue 1 thread](https://github.com/andreacaraffa/freeze/issues/1) and [Issue 2 thread](https://github.com/andreacaraffa/freeze/issues/2) | Requests for code continue into 2026; no later author inference-release URL was found in these inspected threads. |

Related author-controlled sources do not resolve this gap. [fabiopoiesi/gedi](https://github.com/fabiopoiesi/gedi) and [tev-fbk/dGeDi](https://github.com/tev-fbk/dGeDi) provide geometric descriptors, not the complete original FreeZe inference pipeline. The inspected [Flose](https://github.com/tev-fbk/Flose) and [FunFlow6D](https://github.com/tev-fbk/FunFlow6D) repositories are their project websites, not FreeZe V1/V2 source releases.

**Finding:** no verified public author-released inference implementation for FreeZe V1 or FreeZeV2 was located in the checked sources. The official website's Code button must not be treated as evidence that such code is present.

## Third-party implementations

### Selected reference: tfwang08/FreezeV2-Re

This study pins [tfwang08/FreezeV2-Re](https://github.com/tfwang08/FreezeV2-Re) at exact revision **12f82a2f8afd435466a87cf2dafce404087aeb3f**. Its code implements a base pose path, including feature extraction, correspondence matching, RANSAC, ICP, candidate scoring and BOP export. It is an independent reconstruction; no paper-author affiliation or author endorsement was verified.

No code LICENSE/NOTICE file or project license declaration was found at the pinned revision; the GitHub license API returned null. Public visibility is not a redistribution license. Keep the unchanged checkout as an ignored external research dependency, record its revision and hashes, and keep our launcher/reporting code distinct. Do not copy its algorithm files into our tracked implementation or claim an author-code port. Any future redistribution or derivative code publication requires a separately resolved permission/license basis.

The repository does not freeze every parameter needed for a unique published inference result. Its layer, visibility tolerance, raw CAD count and translation-NMS choices remain explicit or unresolved. See REPRODUCTION_NOTES.md. Running this source faithfully can establish replication of that declared third-party configuration; it does not by itself establish reproduction of the paper's score.

### Other inspected V1 reconstruction

[crisiumnih/freeze-pipeline](https://github.com/crisiumnih/freeze-pipeline), inspected main revision `8626b8a3c4316b24a8678977b3745d8938c2a242`, describes itself as an independent implementation from scratch. Its README explicitly lists feature fusion, RANSAC and ICP as not yet implemented. The owner profile identifies Prajwal at SRA-VJTI; no paper-author affiliation was verified. No code license was found. It is not a complete substitute for the required baseline.

A GitHub repository-name search for FreeZeV2 also returned unrelated Lua/game repositories. They were excluded after inspecting their root files.

## Three claims that remain separate

1. Evaluating the authors' untouched prediction CSVs reproduces their published score.
2. Executing the pinned third-party inference produces valid predictions under a fully recorded configuration.
3. Those newly generated predictions reproduce published performance under matching inputs and protocol.

The public repository's downloadable saved FreeZeV2.1 predictions correspond to the **Accurate/challenge reference**, not proof that its own base inference generated those numbers. Successful reference evaluation supports claim 1 only. It must not be reported as claim 2 or 3, nor as evidence that our single-CNOS base has reproduced the Accurate method.