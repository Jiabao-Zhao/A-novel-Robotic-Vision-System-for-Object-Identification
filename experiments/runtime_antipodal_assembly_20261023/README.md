# Training-free grasps: what happened in the first test

**New: [Smaller search on all 14 parts](smaller_search_20261023/README.md).**
The smaller search and fresh larger control each picked up and held 14/14 parts.
The smaller search used 92.4% fewer generated gripper poses and 75.5% less total
search time, retaining the successful +6/+8 mm finger-depth choices. Its
156 shortlisted poses are shown individually. This follow-up tested pickup only.

The first test picked up **13 of 14 parts**, but it did **not demonstrate a
completed assembly**. Two new checks then picked up the 8 mm round pin too: a
center-position check and a fresh automatic search. The original search had
missed a working finger position; the pin is not proven ungraspable.

All of this was simulation with a Panda robot. Part identity and starting
position/orientation were supplied from the simulator, as requested.
Classification and pose-estimation accuracy were not being tested. No physical
robot was connected.

[See the generated grasp pose on every part](../../outputs/runtime_antipodal_assembly_20261023/grasp_pose_gallery/20261024T022411Z/index.html).
Each part has a whole-gripper view and a close-up of its planned finger touches.
The 8 mm round pin is clearly marked as the successful automatic follow-up.

## What the earlier "0/2" meant

That wording was misleading. **Two pin runs started moving downward toward a
hole. Neither pin entered its hole.** The gripper hit fixed hardware on the
board, and the controller stopped before the pin reached the board surface.
This did not measure how accurately either pin could be inserted.

| What actually happened in the first test | Number of parts |
| --- | ---: |
| A separate trial was set up | 14 |
| The robot picked the part up and lifted it at least 12 cm | 13 |
| The robot continued toward an assumed destination on the board | 8 |
| The robot started lowering toward that destination | 7 |
| The robot opened its fingers, withdrew, and recorded the part afterward | 5 |
| A pin actually entered a board hole | 0 |
| A completed assembly was demonstrated | 0 |

The seven lowering moves were three gears, the M16 nut, USB, the 16 mm round pin
and the 12 mm rectangular pin. Only the last two were moves toward pin holes.
The M12 nut stopped earlier, during its move across the table. Five other parts
were deliberately stopped after a successful pickup because the search had not
found a complete route to their goals. The 8 mm round pin was rejected before
any robot motion in the original run.

## Which scene was used

I used the **older 14-part `objects-centered` task-board scene**, saved state
`objects_centered`. I did **not** use one of the recent N00–N07 twelve-part
classification layouts.

Original source, relocated to D: during the October 27 workspace cleanup:

```text
outputs/datasets/nist_task_board/legacy_workspace_inputs/nist-taskboard-workspace/objects-centered/scene.xml
```

The test used a new copy at:

```text
outputs/runtime_antipodal_assembly_20261023/input_scene/scene.xml
```

The copy replaces that old scene's USB-with-cable model, whose contacts were
disabled, with the matching USB CAD and working contact geometry. It also places
USB at a supported starting height. The other 13 parts keep their saved starting
positions and orientations. The original scene was not edited.

![Complete prepared scene before hiding other parts](../../outputs/runtime_antipodal_assembly_20261023/clarification_20261023/full_scene.png)

Each part was tested from a fresh copy of that saved state. **This was not one
continuous operation assembling all 14 parts onto the board.**

## What "other loose parts were excluded" meant

For each trial, I made the other 13 loose parts invisible, disabled their contact
with the robot and target, and held their saved positions fixed. Their simulator
bodies still existed, but they could not block or be pushed by that trial.

Hiding them went beyond the request to ignore their collisions. In the new
8 mm follow-up, they remain visible while their collisions remain disabled.

The table, board and its fixed hardware were kept solid. There was an important
mistake in the assembly setup: **the path search checked the robot and table,
but did not check the board or its fixed hardware. The physics simulation did
check those contacts.** This allowed a path to pass the initial check and then
hit a shaft or screw during movement. The two pin runs stopped this way. The
path search and physical simulation need to use the same fixed obstacles before
this is a useful assembly test.

## What happened to every part

"No robot angles found" below means the numerical solver did not find joint
angles that placed the gripper at the requested point and direction within its
chosen error limits. It does not prove that every possible route is impossible.

| Part | Actual result and specific reason |
| --- | --- |
| Large gear | Picked up, moved over its shaft, lowered and released. It rested on the shaft about **20.00 mm too high**. The last recorded contacts were with that shaft. |
| Medium gear | Completed the same motions, but rested on its shaft about **20.00 mm too high**. |
| Small gear | Completed the same motions, but rested on its shaft about **20.00 mm too high**. |
| Waterproof connector | Picked up and lifted about **180 mm**, then stopped as planned. Of 48 tested full routes, 34 failed to produce robot angles for the move toward the board; 14 had the open fingers hit the target during approach. No physical move to the board was attempted. |
| M12 nut | Picked up and moved partway toward the board. The controller stopped after **86 of 160 transfer steps** because the best gripper position was still **0.518 mm** from its requested position; its solver required **0.1 mm**. No robot/table, robot/board or robot self-contact was recorded. |
| M16 nut | Picked up, moved over its screw, and started lowering. It tipped **before the fingers opened**: by the end of lowering it was tilted **51.6 degrees**, with only one finger touching it. It eventually lay on its side, **21.78 mm from the assumed goal**. |
| 16 mm round pin | Picked up, turned upright, moved above its hole and began lowering. The gripper hit **gear shaft 3**. After **111 of 120 lowering steps**, the next requested joint change exceeded the controller's **8.59 degrees per 0.04-second step** limit. Even its lowest CAD point remained **26.14 mm above the board**. It never entered the hole. |
| 16 mm rectangular pin | Picked up and lifted, then deliberately stopped. All **48** tested full routes failed while turning the held pin. The best gripper position errors were **0.197–0.760 mm**, exceeding the **0.1 mm** position target. |
| 12 mm round pin | Picked up and lifted, then deliberately stopped. There were **120** route checks: 72 failed while turning, 20 while moving toward the board, and 28 would put robot arm link 6 into the table while lowering. The largest table overlap was **8.12 mm**. |
| 8 mm round pin | **Original run:** no robot motion. All **12,936** generated gripper positions failed the table check. **New center-grasp check:** picked up and lifted **159 mm** after raising the gripper **2 mm**; see below. |
| 12 mm rectangular pin | Picked up, turned upright, moved above its hole and began lowering. The gripper hit the **M12 screw**, then arm link 6 touched the board. After **114 of 120 lowering steps**, the next requested joint change exceeded the same **8.59-degree** limit. The pin's lowest CAD point remained **20.78 mm above the board**. It never entered the hole. |
| 8 mm rectangular pin | Picked up and lifted, then deliberately stopped. All **48** tested full routes failed while turning. Position errors were **19.56–20.40 mm** and direction errors **8.13–8.57 degrees**. No physical turning or board move was attempted. |
| USB connector | Picked up, moved to its assumed goal, released and withdrew. Final error was **0.410 mm** and **0.095 degrees**. This goal put USB above the female connector; it did **not** specify plugging it in. In the last one-second check, its largest rotation speed was **3.74 degrees/second**, above the chosen **2.86 degrees/second** stillness limit. |
| D-sub connector | Picked up and lifted, then deliberately stopped. Of 48 tested full routes, 47 failed to produce robot angles for the move toward the board, with position errors **9.32–15.60 mm**. One approach would put the gripper body into the target. No physical board move was attempted. |

For the gears, the likely explanation for stopping on top of the shafts is the
simplified solid shapes used for physical contacts: they close the gear holes.
The 20 mm height difference and recorded shaft contacts support that explanation,
but no separate experiment has isolated that cause. The nut and connector models
also do not yet provide faithful screw-thread or connector-plugging behavior.

## The 8 mm pin: middle of the gripper versus the finger ends

There are two different meanings of "center":

- **Between the two jaws:** the pin was centered between them in both the
  original search and the new check. The new check also used the middle of the
  pin's 50 mm length.
- **Farther up the fingers, toward the middle of their contact pads:** the old
  search included these depths too. With a pin lying only 8 mm high on the
  table, moving it deeper between these particular fingers requires lowering
  more of the finger body into the table.

The original search limited contacts to four positions along the small finger
pads. It did not try a useful contact position on the full finger shape beyond
that range. The best centered position it did try put the finger bodies about
**1.25 mm into the table**, so it was rejected before robot motion.

In the new physical check, moving the gripper **2 mm higher** left about
**0.75 mm** between the fingers and table. The actual finger ends still gripped
the pin, and the robot lifted it **159 mm** with both fingers touching it.
Neither the gripper shape, friction nor contact softness was changed.

Therefore the earlier result means **the original search missed a working
grasp**, not that the 8 mm pin cannot be picked. This new center-position test
was a separate diagnostic, not a rerun of all 14 parts.

![Middle-pad, original and raised finger positions](../../outputs/runtime_antipodal_assembly_20261023/8mm_center_grasp_followup/20261024T011847Z/measured_center_vs_tip.png)

I also reran **automatic grasp generation**, without specifying the contact
points by hand. The change was to allow two more positions along the full finger
shape, 2 mm and 4 mm beyond the old allowed range. Fresh CAD sampling then found
**54 positions that cleared the table**, where the old range found zero. The
first position that also passed the robot checks lifted the pin **160.0 mm**
on its first physical attempt, with both fingers holding it. No robot/table,
robot self-contact or robot/board overlap was recorded. The simulator still
allows small numerical overlap between fingers and part, about 0.59 mm at most
in this follow-up; the contact settings were not softened to get this result.

The other 13 parts remained visible in both new checks, with collisions disabled.
These checks tested pickup only. They did not test inserting the 8 mm pin or
rerun all 14 assembly cases, so the original 13/14 result is kept as recorded.

New evidence:

- [Center-position test and contact measurements](../../outputs/runtime_antipodal_assembly_20261023/8mm_center_grasp_followup/20261024T011847Z/summary.json)
- [Fresh automatic-search result](../../outputs/runtime_antipodal_assembly_20261023/8mm_runtime_followup/20261024T012239Z/result.json)
- [New automatic-search pickup image](../../outputs/runtime_antipodal_assembly_20261023/8mm_runtime_followup/20261024T012239Z/lifted_all_parts_visible.png)
- [Video of the new automatic pickup](../../outputs/runtime_antipodal_assembly_20261023/8mm_runtime_followup/20261024T012239Z/recorded_motion.mp4)
- [Source for the automatic follow-up](clarification_20261023/run_8mm_runtime_followup.py)

## Method, saved evidence and rerunning the test

The [method comparison](METHODS.md) explains why the existing EmbodiChain
GraspKit code was chosen. It finds two opposing places for fingers to touch the
CAD model at runtime. There are no saved hand-picked grasp poses for each part,
no grasp-network training, no weight downloads and no paid calls.

The robot's starting CAD pose and the assumed final poses were supplied. The
controller used robot state while moving; it did not repeatedly read the
object's exact position to correct the grasp. Objects were held by simulated
finger forces, with no invisible attachment holding them to the hand.

Original 14-part run:

```text
outputs/runtime_antipodal_assembly_20261023/20261023T213101Z/
```

The [step-by-step audit](clarification_20261023/failure_phase_audit.json) records
the source files, contacts and errors behind the table above. Raw trajectories,
generated grasps and original score/report files have not been rewritten.
The 11 focused tests and independent record checks passed; that means the code
checks and saved records reproduced, **not that assembly passed**.

Commands in the existing WSL environment, from the repository root:

```bash
PY=/home/jiabao/.venvs/lerobot-libero/bin/python
$PY -m pytest -q experiments/runtime_antipodal_assembly_20261023/test_runtime_grasp.py
$PY -m experiments.runtime_antipodal_assembly_20261023.evaluate_results --self-test
$PY -m experiments.runtime_antipodal_assembly_20261023.run_experiment
```

These commands create new outputs. The first full test sent 2,048 virtual
straight lines toward each CAD surface to find possible finger contacts. It
expanded at most 320 pairs of contacts into gripper positions, checked at most 48
candidate grasps and checked ten positions along each planned movement. Those
limits can miss a usable grasp or route. The original runner is preserved; it
does not silently include the new finger-position search.

To repeat the new automatic 8 mm pickup check:

```bash
$PY -m experiments.runtime_antipodal_assembly_20261023.clarification_20261023.run_8mm_runtime_followup
```

The pose interface takes `world_T_cad`, a 4-by-4 matrix giving the CAD's position
and orientation in the simulated world, in metres. The current finger and robot
checks use Panda geometry. The real UR/RG2 robot needs its own finger geometry,
tool reference point and robot model before this can be used on that hardware.
