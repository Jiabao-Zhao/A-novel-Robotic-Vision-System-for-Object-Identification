from pathlib import Path


EXPERIMENT_NAME = "put_all_objects_into_basket"
EXPERIMENT_ROOT = Path("outputs/simulation/experiments") / EXPERIMENT_NAME
VLA_METHOD_FOLDER = "VLA"
PROPOSED_METHOD_FOLDER = "proposed_framework"
METHOD_FOLDERS = (VLA_METHOD_FOLDER, PROPOSED_METHOD_FOLDER)

LIBERO_OBJECT_TASKS = (
    (0, "alphabet_soup", "pick up the alphabet soup and place it in the basket"),
    (1, "cream_cheese", "pick up the cream cheese and place it in the basket"),
    (2, "salad_dressing", "pick up the salad dressing and place it in the basket"),
    (3, "bbq_sauce", "pick up the bbq sauce and place it in the basket"),
    (4, "ketchup", "pick up the ketchup and place it in the basket"),
    (5, "tomato_sauce", "pick up the tomato sauce and place it in the basket"),
    (6, "butter", "pick up the butter and place it in the basket"),
    (7, "milk", "pick up the milk and place it in the basket"),
    (8, "chocolate_pudding", "pick up the chocolate pudding and place it in the basket"),
    (9, "orange_juice", "pick up the orange juice and place it in the basket"),
)


def episode_result_dir(method_folder, task_index, initial_state_index):
    if method_folder not in METHOD_FOLDERS:
        raise ValueError(
            f"Unknown experiment method folder {method_folder!r}; "
            f"expected one of {METHOD_FOLDERS}."
        )
    task = next(
        (item for item in LIBERO_OBJECT_TASKS if item[0] == int(task_index)),
        None,
    )
    if task is None:
        raise ValueError(f"Unknown LIBERO-Object task index: {task_index}")
    if int(initial_state_index) < 0:
        raise ValueError("Initial-state index must be non-negative.")
    return (
        EXPERIMENT_ROOT
        / method_folder
        / f"task_{task[0]:02d}_{task[1]}"
        / f"init_state_{int(initial_state_index):02d}"
    )
