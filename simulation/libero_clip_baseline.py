"""Optional local CLIP grounding baseline; not used by the primary task runner."""

import numpy as np
from PIL import Image


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
LIBERO_OBJECT_LABELS = (
    "a red carton of milk",
    "a woven basket",
    "a can of tomato sauce",
    "a package of cream cheese",
    "a package of butter",
    "a bottle of orange juice",
    "a cup of chocolate pudding",
)


def ground_milk_and_basket(rgb, localized_objects, model_name=CLIP_MODEL_NAME):
    """Ground the task roles from RGB crops without simulator object metadata."""
    try:
        import torch
        from transformers import AutoProcessor, CLIPModel
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "CLIP grounding needs the torch and transformers packages supplied "
            "by the LeRobot environment."
        ) from error

    objects = list(localized_objects)
    if len(objects) < 2:
        raise RuntimeError("At least two localized candidates are required for pick and place.")
    crops = [_crop_object(rgb, item["roi"]) for item in objects]

    try:
        model = CLIPModel.from_pretrained(model_name, local_files_only=True)
        processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
    except OSError as error:
        raise RuntimeError(
            f"CLIP checkpoint {model_name!r} is not cached. Download it once with "
            f"`python -c \"from transformers import CLIPModel, AutoProcessor; "
            f"CLIPModel.from_pretrained('{model_name}'); "
            f"AutoProcessor.from_pretrained('{model_name}')\"`."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    image_inputs = processor(images=crops, return_tensors="pt")
    text_inputs = processor(text=list(LIBERO_OBJECT_LABELS), padding=True, return_tensors="pt")
    image_inputs = {name: value.to(device) for name, value in image_inputs.items()}
    text_inputs = {name: value.to(device) for name, value in text_inputs.items()}
    with torch.inference_mode():
        outputs = model(**image_inputs, **text_inputs)
        similarities = outputs.image_embeds @ outputs.text_embeds.T

    scores = similarities.detach().cpu().numpy()
    milk_index = int(np.argmax(scores[:, 0]))
    basket_index = int(np.argmax(scores[:, 1]))
    if milk_index == basket_index:
        raise RuntimeError("CLIP assigned milk and basket to the same localized candidate.")

    evaluations = []
    for item, row in zip(objects, scores):
        label_index = int(np.argmax(row))
        evaluations.append(
            {
                "object_id": item["object_id"],
                "best_label": LIBERO_OBJECT_LABELS[label_index],
                "best_similarity": float(row[label_index]),
                "milk_similarity": float(row[0]),
                "basket_similarity": float(row[1]),
            }
        )

    return {
        "model": model_name,
        "device": device,
        "method": "CLIP over RGB crops from depth-localized candidates",
        "milk_object_id": objects[milk_index]["object_id"],
        "basket_object_id": objects[basket_index]["object_id"],
        "milk_similarity_margin": _selection_margin(scores[:, 0]),
        "basket_similarity_margin": _selection_margin(scores[:, 1]),
        "evaluations": evaluations,
    }


def _crop_object(rgb, roi, padding_px=8):
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    x1 = max(0, int(roi["x1"]) - padding_px)
    y1 = max(0, int(roi["y1"]) - padding_px)
    x2 = min(image.width, int(roi["x2"]) + padding_px)
    y2 = min(image.height, int(roi["y2"]) + padding_px)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError(f"Invalid localized ROI for CLIP grounding: {roi}")
    return image.crop((x1, y1, x2, y2))


def _selection_margin(scores):
    ordered = np.sort(np.asarray(scores, dtype=float))
    return float(ordered[-1] - ordered[-2])
