# Why sampled finger-end grasps were rejected

A first-contact fit is a prerequisite. If it does not produce both finger contacts, later table/receiver/protected checks are not evaluated. Contact-region failures include failed first-contact fits. Later checks cover only fitted poses. Counts overlap: one pose can fail several checks. No physical pickup was attempted.

| Part | Fitted / 800 | Valid / 800 | Contact-direction failures | Contact-region failures | Part overlap | Table | Receiver | Protected surface |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BNC Male | 296 | 7 | 200 | 584 | 0 | 268 | 104 | 0 |
| DSUB Male | 32 | 6 | 0 | 768 | 0 | 26 | 0 | 0 |
| Gear Large | 360 | 4 | 344 | 784 | 24 | 232 | 184 | 16 |
| KET12 Square 12mm | 40 | 13 | 0 | 760 | 0 | 24 | 8 | 0 |
| KET16 Square 16mm | 24 | 12 | 0 | 776 | 0 | 12 | 0 | 0 |
| KET8 Square 8mm | 64 | 16 | 0 | 736 | 0 | 40 | 24 | 0 |
| M16 Hex Nut | 104 | 9 | 0 | 696 | 0 | 92 | 80 | 0 |
| RGOCG12-50 12mm | 120 | 0 | 120 | 800 | 0 | 98 | 64 | 0 |
| RGOCG16-50 16mm | 64 | 1 | 56 | 792 | 0 | 52 | 40 | 0 |
| RGOCG8-50 8mm | 144 | 0 | 144 | 800 | 0 | 129 | 80 | 0 |
| USB Male | 120 | 0 | 112 | 784 | 0 | 120 | 32 | 0 |
| Waterproof Male | 64 | 7 | 40 | 768 | 0 | 47 | 0 | 0 |

## Round 8 mm pin

First-contact outcomes across 800 proposals: {'first_contact_not_on_distal_finger': 656, 'fitted': 144}.

N00: 18 fitted; check outcomes {'candidate_input': {'pass': 100}, 'opening_limits': {'not_evaluated': 82, 'pass': 18}, 'contact_reconstruction': {'not_evaluated': 82, 'pass': 18}, 'surface_normals': {'not_evaluated': 82, 'fail': 18}, 'pad_contact_geometry': {'fail': 100}, 'target_closing_collision': {'not_evaluated': 82, 'pass': 18}, 'table_clearance': {'not_evaluated': 82, 'fail': 18}, 'final_receiver_collision': {'not_evaluated': 82, 'fail': 10, 'pass': 8}, 'protected_surfaces': {'not_evaluated': 100}}.
N05: 18 fitted; check outcomes {'candidate_input': {'pass': 100}, 'opening_limits': {'not_evaluated': 82, 'pass': 18}, 'contact_reconstruction': {'not_evaluated': 82, 'pass': 18}, 'surface_normals': {'not_evaluated': 82, 'fail': 18}, 'pad_contact_geometry': {'fail': 100}, 'target_closing_collision': {'not_evaluated': 82, 'pass': 18}, 'table_clearance': {'not_evaluated': 82, 'fail': 5, 'pass': 13}, 'final_receiver_collision': {'not_evaluated': 82, 'fail': 10, 'pass': 8}, 'protected_surfaces': {'not_evaluated': 100}}.

The separate declared straight-down side grip clears the table by 2.238 mm after the contact correction, versus -1.103 mm before. Its 1.107-degree contact error still exceeds the unchanged 1-degree limit, and its nominal final hand pose intersects the board. It is a diagnostic pose outside the random bank, not an additional selected grasp.

The N00 100-proposal pilot independently checked each failed fit: all 82 had the saved pad touch before the finger body. This was a static geometry probe of those same proposals, without a retry bank. That detailed finding should not automatically be assigned to every failed fit for other products.

The model uses original convex finger collision surfaces. Its central visual groove is not resolved by those solids; hardware contact remains unverified.
