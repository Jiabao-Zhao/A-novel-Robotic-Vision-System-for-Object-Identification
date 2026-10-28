FINGER_BUFFER = 0.25  # buffer distance for robot finger collision checking (cm)
HAND_KNUCKLE_BUFFER = 1.0  # buffer distance for robot hand knuckle collision checking (cm)
ARM_BUFFER = 1.0  # buffer distance for robot arm collision checking (cm)

RETRACT_OPEN_RATIO = 0.1  # extra open ratio for retract grasp
RETRACT_DELTA_NEAR = 1.0 # incremental distance for retract grasp when near
RETRACT_DELTA_FAR = 5.0 # incremental distance for retract grasp when far away

CHECK_GRIPPERS_INTERLOCK = False # whether to check grippers interlock during dual-arm planning

OPEN_RATIO_REST = 0.5 # open ratio for resting position
