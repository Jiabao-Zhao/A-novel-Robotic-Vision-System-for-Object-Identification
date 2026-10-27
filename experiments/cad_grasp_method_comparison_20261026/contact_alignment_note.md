# Contact alignment and movement during finger closure

The user's concern is valid: a pair that can lift the object may still push or turn it while the fingers close. Favoring finger pads that face the local surfaces squarely is appropriate for the assembly goal. It reduces one cause of sliding; it does not establish zero displacement. Zhao et al. explicitly separate successful pickup from post-grasp displacement for top-down parallel-jaw assembly. Their method uses trained networks; the citation supports the distinction, not adopting those networks here. [Zhao et al., introduction and section 3.1](https://proceedings.mlr.press/v155/zhao21c/zhao21c.pdf).

Three different conditions matter:

1. **Direction:** each pad should face its contacted surface squarely. On a flat face this can provide broad contact. On a smooth round side a flat pad can instead touch along a tangent line. Round pins and BNC side grips should remain eligible when that local direction is well aligned; requiring two globally planar CAD faces would remove them unnecessarily.
2. **Position:** the opposing forces should act along compatible lines. Even equal opposite forces can turn the part when their lines are offset. With actual force locations `pL`, `pR` and equal opposite forces `f`, the turning moment is `(pL-pR) × f`. The selected CAD witness points are not measured force locations or contact-pressure centers, so their separation alone is not a proven turning-moment estimate.
3. **First touch:** one finger may reach a shoulder, tooth, edge or curved section before its intended contact. Until the other finger provides support, closure can move the object. The finite-hand checks and modeled first-contact centering address this geometric risk. Closing gently reduces impact but cannot guarantee that an offset contact will leave the part stationary.

Kleeberger et al. combine orientation, friction and collision conditions, but explicitly state that their generator does not model physical interaction between the object and gripper. Their collision model also allows slight surface unevenness or curvature. Thus a geometric acceptance result should remain separate from a measured no-movement result. [Kleeberger et al., sections 3.1 and 4.3](https://arxiv.org/html/2104.11660v1).

## What a one-degree test can claim

A controlled test requiring both saved first-contact triangle normals to be within **1 degree** of the closing direction directly tightens the local alignment measurement. It is a declared trial setting chosen for this experiment, not a cited universal threshold or a validated displacement limit. Keep the existing finite-pad, target/table and bilateral first-contact checks. Do not describe this screen as proof of full-pad flat contact, force balance, unchanged part-to-gripper pose or assembly precision.

There are important implementation distinctions:

- In the frozen pool, `core.validate` collects simultaneous inner-pad contacts whose normals pass **15 degrees**, then saves the one nearest the pad center. Filtering that saved witness at 1 degree is not identical to selecting a witness again from the simultaneous set with a 1-degree limit. Another qualifying witness may exist but was not saved. Label the former precisely as a **saved-witness screen**.
- The native sampled contact line may still have been projected by up to 15 degrees to construct a downward hand. A 1-degree actual-surface-normal screen does not change or erase that separate conversion allowance.
- Native triangle normals represent a faceted mesh. On curved parts, a triangle normal can differ from the smooth cylinder's exact tangent direction. A 1-degree limit can therefore remove a useful round-side grasp because of mesh resolution. Preserve native normals for the audit; do not silently replace them with smoothed normals and claim the same test.
- The existing approximately simultaneous contact condition compares the modeled finger openings at first touch, with a 50 micrometer tolerance. It is not measured contact timing, force symmetry or physical accuracy.

The appropriate claim after the controlled test is: **these retained candidates satisfy a stricter modeled local contact-alignment requirement while preserving eligible round-side grips**. Whether this reduces motion during closure remains unmeasured. No new quality score or robot execution is required to make that limited geometric claim.
