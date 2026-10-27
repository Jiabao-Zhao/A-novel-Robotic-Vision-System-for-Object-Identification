# Fixed-proposal shared-contact-region ablation

All518 frozen hand proposals from200202Z are held fixed. The real modeled Panda hand, target triangles, observed poses/table and core thresholds are unchanged. No new grasp is generated and no pickup is attempted.

| Condition | Passing checked poses | Placements with at least one |
|---|---:|---:|
| saved_strict | 359 | 87/96 |
| strict_25 | 359 | 87/96 |
| band_50 | 351 | 87/96 |
| band_100 | 353 | 87/96 |
| band_250 | 353 | 87/96 |
| normal_screen_only | 415 | 96/96 |
| witness_shared_25 | 363 | 88/96 |
| witness_no_shared | 389 | 92/96 |
| core_only | 481 | 96/96 |

Strict25 exactly reproduces every frozen decision. Core-only independently replays all518 original full-hand checks.

The paired-point25/50/100/250µm tests change only the geometric proximity band used to find common regions; all later paired-point normal/shaft/radial checks still run. Largerbands do not modify robot solids, object triangles, actual first contacts or the core interference check.

The matched witness_shared25 and witness_no_shared tests use identical actual first-touch witnesses, individual guarded pad-region existence, normal opposition, cylinder membership and radial/mesh checks. Only requiring sharedY/Z differs between those two columns. This is a clean guard-effect comparison under that witness formulation. Comparing either witness condition directly with the original strict paired-point test also changes where the later radial checks are evaluated; it is not described as pure guard removal.

normal_screen_only retains the added1°flat/15°classified-cylinder native-normal test without the common-region or fitted-radial screens. It isolates how much rejection already occurs before shared-region computation.

| Original missing placement | Proposals | Strict25 | Normal only | Witness+shared | Witness without shared | Core only |
|---|---:|---:|---:|---:|---:|---:|
| N00/BNC_Male | 8 | 0 | 6 | 0 | 2 | 8 |
| N01/BNC_Male | 8 | 0 | 4 | 0 | 0 | 8 |
| N01/Gear_Large | 8 | 0 | 8 | 0 | 8 | 8 |
| N03/BNC_Male | 8 | 0 | 4 | 0 | 0 | 8 |
| N03/Gear_Large | 8 | 0 | 6 | 0 | 6 | 8 |
| N04/BNC_Male | 6 | 0 | 6 | 0 | 0 | 6 |
| N05/Gear_Large | 4 | 0 | 4 | 4 | 4 | 4 |
| N06/BNC_Male | 8 | 0 | 4 | 0 | 0 | 8 |
| N07/Gear_Large | 8 | 0 | 4 | 0 | 4 | 8 |

The two levels of failure matter: preparation had excluded some contact regions before a hand pose existed; this fixed-proposal test cannot recover those regions. Every listed proposal did exist in the frozen run. Removed screens increase modeled candidate availability, not demonstrated pickup success or assembly precision.

Replay cost with two CPU workers, not fresh generation time. Native geometry/table/hand and all518proposalposes remain fixed. Largerbands are sensitivity experiments, not calibrated deformation or acceptableassemblyoffset.
