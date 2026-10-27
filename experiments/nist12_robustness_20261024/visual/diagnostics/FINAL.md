# Final Qwen description health

The final input list contains **4,730 images and 9,460 description rows**.
There are 121 unavailable rows: 120 crop runtime errors and the unchanged
`TLESS_000007` material-tag parse failure described in the first-snapshot report.
No accuracy annotation was read, and no failed request was retried.

**The fourteen newly captured HTTP 400 response bodies explicitly establish
context overflow for those requests.** Their reported request lengths are
4,224, 4,253, 4,330 or 4,359 tokens against a 4,096-token available context.
For example, the saved server response is:

> request (4359 tokens) exceeds the available context size (4096 tokens), try increasing it

Those fourteen requests cover seven distinct image hashes and appear in 36 final
description rows through exact-image reuse. The original 32 failed request files
still lack response bodies; they account for 84 final rows after further reuse.
Their exact cause remains unknown. Similar crop size and conditions support an
association but do not retroactively prove the server's reason for the old errors.
Every original failed raw file still matches its pre-change SHA256.

Across both passes, 46 failed request files cover 23 distinct image hashes and
sixty crop inputs. All sixty Gaussian-depth-noise scenes have one failed crop:
twenty scenes each at 1, 2 and 5 mm noise. Fifty-five of those scenes have no
other description-available crop, blocking description evidence for **660 CAD
queries**. Five scenes retain other description-available candidates, which can
proceed under the frozen rank-available rule. The one malformed CAD-view
description independently blocks **twenty real-pilot queries** for that CAD.

The final description-stage total is therefore **680 blocked queries out of
5,000 requested-CAD tasks**. This is an availability count, **not classification
accuracy**, a false-negative count, or a claim that the target is absent. It
precedes CLIP-vector validity checks and does not affect depth-only or size-only
selection, which do not consume Qwen descriptions.

Evidence is in `final_description_health.json`, `final_http_error_bodies.json`
and `final_unavailable_impact.json`. The exact response-body bytes remain in
their original raw records, with Base64, UTF-8 text, SHA256 and byte length.
The diagnostic inventory is bound to the final input/description SHA256 values.
No prompt, image, model, generation setting, parser or existing raw output was
changed to recover these failures.
