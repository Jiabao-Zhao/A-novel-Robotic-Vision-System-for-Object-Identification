# Observation-based board retry

This separate trial preserves the first run and its seven runtime files. It
repeats fresh camera-based localization and physical simulated pickups for the
large gear, medium gear and M16 nut. It does not start from saved object poses.

Two specific first-run problems motivate the changes:

- The medium gear's direct turn failed the robot route check. Search a small set
  of raised and staged turns, checking every candidate with the known robot and
  board model before moving. Keep the original requested CAD goal orientation.
- Board surfaces entered the large gear's local depth fitting region. Predict
  depth from the known robot, table and board at measured robot joint positions.
  Remove raw depth pixels agreeing with that prediction within 1 mm. This
  prediction has no product body, estimated product pose or simulator state.
  Raw images, removed pixels and hashes remain available for review.

The CAD fitting code and its acceptance limits are unchanged. Removing known
surfaces can also remove a part surface within 1 mm of the board; unavailable
estimates still stop movement. The robot observes again after pickup, turning,
transfer and each lowering step. Each new part-to-gripper transform uses the
observed CAD pose and simultaneous robot forward kinematics.

Use `python -m experiments.observed_held_assembly_20261025.board_feedback_followup_20261025.run_complete_responses`
inside the existing WSL experiment environment. Outputs go to a new timestamped
folder under `outputs/observed_held_assembly_20261025/board_feedback_followup_20261025/runs`.

The first follow-up (`20261025T202719Z`) exposed an incomplete file response
while the large gear was approaching. Its result is preserved as a technical
failure, with no large-gear finger-closing pickup. The separate launcher waits
for a complete JSON response; it never resends a robot command. A focused test
deliberately splits a response write and checks that only one request exists.

The destinations are the previously approved nominal CAD goals. Full CAD-frame
rotation, position and positive CAD-Z direction are checked before release.
Local depth fitting does not prove unique keyway, tooth or thread orientation.
Gear bores and nut threads are not represented by the current physical contact
model, so reaching a pose would still not verify mechanical engagement.

This uses simulated raw RGB-D and robot feedback to test the observation
boundary needed for later physical work. It is not a physical robot test.
