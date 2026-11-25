#!/usr/bin/env python3
"""YOLOv8 wing-cell detection and image-orientation preparation.

The detector is deliberately independent of SAM3.  It returns one native
xyxy box per required cell, ready for ``Sam3TrackerProcessor(input_boxes=...)``.
``ultralytics`` and Pillow are imported lazily so the selection logic can be
unit-tested without installing the inference stack on the host.
"""

import json
import os

YOLO_WEIGHTS = "/app/models/yolo_wings_best.pt"
YOLO_IMAGE_SIZE = 1024
YOLO_CONFIDENCE = 0.5

# Fixed training IDs.  masks5/masks9 are the legacy downstream cell names;
# the current CSV and QC files use pa2r/dm respectively.
CLASS_TO_CELL = {0: "pa2r", 1: "dm"}
CLASS_TO_MASK = {0: "masks5", 1: "masks9"}
CELL_TO_CLASS = {cell: class_id for class_id, cell in CLASS_TO_CELL.items()}
REQUIRED_CELLS = ("pa2r", "dm")


def log(message):
    print(message, flush=True)


class IncompleteDetectionError(RuntimeError):
    """A clean, actionable detection failure (not an inference traceback)."""

    def __init__(self, image_name, boxes):
        self.image_name = os.path.basename(image_name)
        self.boxes = boxes
        self.missing = [cell for cell in REQUIRED_CELLS if cell not in boxes]
        found = ", ".join(sorted(boxes)) or "none"
        missing = ", ".join(self.missing)
        super().__init__(
            f"YOLOv8 detection incomplete for '{self.image_name}': "
            f"found {found}; missing {missing} at conf >= {YOLO_CONFIDENCE}."
        )


def require_complete_boxes(image_name, boxes):
    """Implement the required len(boxes) < 2 guard without crashing."""
    if len(boxes) < 2:
        log(f"WARNING: {IncompleteDetectionError(image_name, boxes)}")
        log(f"Skipping automatic segmentation for '{os.path.basename(image_name)}'. "
            "Select manual prompt fallback for this image.")
        raise IncompleteDetectionError(image_name, boxes)
    return boxes


def load_aligned_image(path):
    """Return an RGB image in the upright-landscape training orientation.

    Cleanup images used for training are upright and landscape.  EXIF is
    applied first (phone photos commonly store rotation only in metadata).
    A portrait result is rotated 90 degrees counter-clockwise; left/right
    mirroring is already represented in training (fliplr=0.5).  The SAME
    aligned image must be used for YOLO, SAM3 and QC so all coordinates share
    one pixel frame.
    """
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        exif_orientation = source.getexif().get(274, 1)
        original_size = list(source.size)
        image = ImageOps.exif_transpose(source).convert("RGB")

    exif_size = list(image.size)
    portrait_rotation = 0
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_90)
        portrait_rotation = 90

    info = {
        "source": os.path.basename(path),
        "original_size": list(original_size),
        "exif_orientation": exif_orientation,
        "size_after_exif": exif_size,
        "portrait_rotation_ccw": portrait_rotation,
        "aligned_size": list(image.size),
        "color_mode": "RGB",
        "training_orientation": "upright landscape",
    }
    return image, info


class WingCellDetector:
    """Load YOLO once, then select the best box for each required class."""

    def __init__(self, weights_path=YOLO_WEIGHTS):
        if not os.path.isfile(weights_path):
            raise RuntimeError(f"YOLOv8 weights not found: {weights_path}")
        from ultralytics import YOLO

        # Exactly one model load per segmentation-container startup.
        self.weights_path = weights_path
        self.model = YOLO(weights_path)

    def detect(self, image):
        """Predict at the training resolution and return {cell: detection}.

        Every ``xyxy`` value is [x_min, y_min, x_max, y_max] in pixels of the
        aligned input image.  Duplicate class IDs are resolved by confidence.
        Missing classes are returned as missing keys for the caller's clean
        fallback path.
        """
        results = self.model.predict(
            image, imgsz=YOLO_IMAGE_SIZE, conf=YOLO_CONFIDENCE, verbose=False
        )
        boxes = {}
        if not results:
            return boxes

        predictions = results[0].boxes
        if predictions is None:
            return boxes

        for box in predictions:
            class_id = int(box.cls[0].cpu().item())
            if class_id not in CLASS_TO_CELL:
                continue
            cell = CLASS_TO_CELL[class_id]
            confidence = float(box.conf[0].cpu().item())
            if cell in boxes and confidence <= boxes[cell]["confidence"]:
                continue
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            boxes[cell] = {
                "class_id": class_id,
                "cell": cell,
                "mask_target": CLASS_TO_MASK[class_id],
                "xyxy": xyxy,
                "confidence": confidence,
            }
        return boxes


def write_detection_json(path, image_name, aligned_image_name, alignment,
                         boxes, weights_path=YOLO_WEIGHTS):
    """Persist complete or partial detections for audit/QC and fallback."""
    document = {
        "image": os.path.basename(image_name),
        "aligned_image": (os.path.basename(aligned_image_name)
                          if aligned_image_name else None),
        "alignment": alignment,
        "model": weights_path,
        "predict": {"imgsz": YOLO_IMAGE_SIZE, "conf": YOLO_CONFIDENCE},
        "class_mapping": {
            "0": {"cell": "pa2r", "mask_target": "masks5"},
            "1": {"cell": "dm", "mask_target": "masks9"},
        },
        "boxes": boxes,
        "complete": len(boxes) == 2,
    }
    with open(path, "w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")
    return document
