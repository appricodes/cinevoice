"""
Per-person body-part segmentation for a single video frame, using DensePose.

The model is the standalone TorchScript export of DensePose R-50-FPN-DeepLab
(https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image, Apache-2.0).
It carries the whole detectron2 pipeline inside the .torchscript archive, so nothing
beyond torch/torchvision needs to be installed -- detectron2 itself does not build on
Windows.

What `analyze()` returns is two integer maps the size of the frame:

    part_id[y, x]    0 = background, 1..24 = DensePose body-surface part
    person_id[y, x]  0 = background, 1..N  = which detected person

so a mouse position maps straight to "person 2, left forearm" with a single array
lookup -- see `PartMap.describe()`. Inference is the slow part (seconds on CPU), the
lookup is free, so analyze one paused frame and keep the result.
"""
import os
import threading

import cv2
import numpy as np

# torch is imported inside the methods that need it, not here: it costs a second or two
# to import, and the player should only pay that the first time a frame is analysed.

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "densepose", "densepose_r50_fpn_dl.torchscript",
)

# DensePose splits the body surface into 24 charts. Most limbs get two chart indices,
# one for the front face and one for the back, which is a distinction nobody wants in a
# tooltip -- so both indices map to the same name here. Index 0 is background.
#
# Left/right is the *subject's* own left/right. The two indices for the feet (5, 6) are
# reported in opposite order by different published copies of this table, so verify that
# pair against a real frame before trusting it; the part index in the tooltip makes that
# easy to check.
PART_NAMES = {
    0: None,
    1: "تن", 2: "تن",
    3: "دست راست",
    4: "دست چپ",
    5: "کف پای راست",
    6: "کف پای چپ",
    7: "ران راست", 9: "ران راست",
    8: "ران چپ", 10: "ران چپ",
    11: "ساق راست", 13: "ساق راست",
    12: "ساق چپ", 14: "ساق چپ",
    15: "بازوی چپ", 17: "بازوی چپ",
    16: "بازوی راست", 18: "بازوی راست",
    19: "ساعد چپ", 21: "ساعد چپ",
    20: "ساعد راست", 22: "ساعد راست",
    23: "سر", 24: "سر",
}

PART_NAMES_EN = {
    0: None,
    1: "torso", 2: "torso",
    3: "right hand", 4: "left hand",
    5: "right foot", 6: "left foot",
    7: "right thigh", 9: "right thigh",
    8: "left thigh", 10: "left thigh",
    11: "right leg", 13: "right leg",
    12: "left leg", 14: "left leg",
    15: "left upper arm", 17: "left upper arm",
    16: "right upper arm", 18: "right upper arm",
    19: "left forearm", 21: "left forearm",
    20: "right forearm", 22: "right forearm",
    23: "head", 24: "head",
}

N_PART_LABELS = 24

# The smallest area, in pixels of the analysed frame, still worth announcing as its own
# region. Below this a part is a sliver at the edge of a limb that the pointer can barely
# land on, and one more object for a screen reader to step through for nothing.
MIN_REGION_PIXELS = 60


def _is_persian(language):
    text = (language or "").strip().lower()
    return "persian" in text or "farsi" in text or "فارسی" in text


def names_for_language(language):
    """The part-name table matching the app's current description language."""
    return PART_NAMES if _is_persian(language) else PART_NAMES_EN


def person_word_for_language(language):
    """The word before the person's number, in the same language as the part names."""
    return "نفر" if _is_persian(language) else "Person"


class Region:
    """One announceable area: a named body part of one person, with its own pixel mask."""

    def __init__(self, person, name, mask, bbox):
        self.person = person      # 1-based, numbered left to right across the frame
        self.name = name          # "left forearm" / "ساعد چپ"
        self.mask = mask          # (H, W) bool, true inside this part
        self.bbox = bbox          # (x0, y0, x1, y1) in frame pixels
        self.area = int(mask.sum())

    def label(self, person_word="Person"):
        # Persian takes its own comma; a Latin one is read out as a stray mark by some
        # screen readers rather than as the pause it is meant to be.
        separator = "، " if person_word == "نفر" else ", "
        return "{} {}{}{}".format(person_word, self.person, separator, self.name)


class PartMap:
    """The result of analysing one frame: two label maps plus the person boxes."""

    def __init__(self, part_id, person_id, boxes, size):
        self.part_id = part_id        # (H, W) uint8, 0..24
        self.person_id = person_id    # (H, W) uint8, 0..N
        self.boxes = boxes            # list of (x0, y0, x1, y1) in frame pixels
        self.size = size              # (width, height) of the frame these maps match

    @property
    def n_people(self):
        return len(self.boxes)

    def at(self, x, y):
        """(person_index, part_index) at a frame pixel; person_index 0 means background."""
        w, h = self.size
        if not (0 <= x < w and 0 <= y < h):
            return 0, 0
        return int(self.person_id[y, x]), int(self.part_id[y, x])

    def describe(self, x, y, names=PART_NAMES, show_index=True):
        """Tooltip text for a frame pixel, or None when the pixel is not on a person."""
        person, part = self.at(x, y)
        if not person or not part:
            return None
        name = names.get(part) or "بخش {}".format(part)
        if show_index:
            return "نفر {} — {} ({})".format(person, name, part)
        return "نفر {} — {}".format(person, name)

    def regions(self, names=PART_NAMES_EN, min_pixels=MIN_REGION_PIXELS):
        """
        The frame split into named areas, one per (person, body part).

        DensePose gives most limbs two chart indices, front and back, which share a name;
        they are merged here, so hovering anywhere on a forearm gives one answer however
        the arm is turned. Areas never overlap -- every pixel has exactly one part and one
        person -- which is what lets each become its own object for a screen reader.
        Largest first, so a caller can add them in an order that leaves small parts on top.
        """
        out = []
        for person in range(1, self.n_people + 1):
            on_person = self.person_id == person
            if not on_person.any():
                continue
            seen = []
            for index in range(1, N_PART_LABELS + 1):
                name = names.get(index)
                if not name or name in seen:
                    continue
                seen.append(name)
                indices = [i for i, n in names.items() if n == name]
                mask = on_person & np.isin(self.part_id, indices)
                if int(mask.sum()) < min_pixels:
                    continue
                ys, xs = np.where(mask)
                bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                out.append(Region(person, name, mask, bbox))
        # A person whose every part fell under min_pixels -- someone tiny in the far
        # background -- contributes no area at all. Close the gap their number would leave,
        # so what gets announced runs 1, 2, 3 with nobody silently missing in between.
        present = sorted({r.person for r in out})
        renumber = {old: new for new, old in enumerate(present, start=1)}
        for region in out:
            region.person = renumber[region.person]

        out.sort(key=lambda r: r.area, reverse=True)
        return out

    def overlay(self, frame_bgr, alpha=0.55):
        """The frame with each body part tinted, for a visual check of the maps."""
        colored = cv2.applyColorMap(
            (self.part_id.astype(np.float32) * (255.0 / N_PART_LABELS)).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS,
        )
        mask = (self.part_id > 0)[:, :, None]
        blended = np.where(mask, cv2.addWeighted(frame_bgr, 1 - alpha, colored, alpha, 0), frame_bgr)
        return blended.astype(np.uint8)


class DensePoseParts:
    """Loads the TorchScript DensePose model once and runs it on single frames."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, device="cpu", detect_size=640):
        self.model_path = model_path
        self.device = device
        # The model resizes internally to its training resolution (shortest edge 800),
        # which is slow on CPU. Feeding it a smaller frame and scaling the label maps
        # back up is several times faster and costs almost nothing in tooltip accuracy.
        self.detect_size = detect_size
        self._model = None
        self._lock = threading.Lock()

    def available(self):
        return os.path.isfile(self.model_path)

    def load(self):
        """Load the model, if it is not already loaded. Safe to call from any thread."""
        import torch
        # The archive calls the torchvision::nms custom op, which is only registered as a
        # side effect of importing torchvision. Without this import torch.jit.load fails
        # outright with "Unknown builtin op: torchvision::nms".
        import torchvision  # noqa: F401

        with self._lock:
            if self._model is None:
                if not self.available():
                    raise FileNotFoundError(
                        "DensePose model not found at {}. Download "
                        "densepose_r50_fpn_dl.torchscript from "
                        "https://huggingface.co/LayerNorm/DensePose-TorchScript-with-hint-image"
                        .format(self.model_path)
                    )
                model = torch.jit.load(self.model_path, map_location=self.device)
                model.eval()
                self._model = model
        return self._model

    def analyze(self, frame_bgr):
        """Run DensePose on one BGR frame and return a PartMap at the frame's own size."""
        import torch
        import torch.nn.functional as F

        model = self.load()
        h, w = frame_bgr.shape[:2]

        scale = min(1.0, float(self.detect_size) / max(h, w))
        if scale < 1.0:
            small = cv2.resize(
                frame_bgr,
                (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = frame_bgr
            scale = 1.0
        sh, sw = small.shape[:2]

        tensor = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).to(self.device)
        with torch.no_grad():
            boxes, coarse_segm, fine_segm, _u, _v = model(tensor)

        part_map = np.zeros((sh, sw), dtype=np.uint8)
        person_map = np.zeros((sh, sw), dtype=np.uint8)
        boxes = boxes.cpu()

        # The detector also emits a few degenerate boxes clipped to nothing at the frame
        # edge. Decode every instance first and keep only those that cover real pixels,
        # so person ids stay 1..n_people with no gaps and no phantom people.
        decoded = []
        for i in range(len(boxes)):
            x0, y0, x1, y1 = [int(round(float(v))) for v in boxes[i]]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(sw, x1), min(sh, y1)
            bw, bh = x1 - x0, y1 - y0
            if bw < 2 or bh < 2:
                continue
            coarse = F.interpolate(coarse_segm[i:i + 1].float(), (bh, bw),
                                   mode="bilinear", align_corners=False)
            fine = F.interpolate(fine_segm[i:i + 1].float(), (bh, bw),
                                 mode="bilinear", align_corners=False)
            body = coarse.argmax(dim=1) > 0
            labels = (fine.argmax(dim=1) * body)[0].to(torch.uint8).cpu().numpy()
            if not labels.any():
                continue
            decoded.append((i, (x0, y0, x1, y1), labels))

        # Number people left to right across the frame, not by detection score: "person 1"
        # then means the same person however the scores come out, and a listener can match
        # the numbering to how the scene was described to them.
        left_to_right = sorted(range(len(decoded)),
                               key=lambda k: (decoded[k][1][0] + decoded[k][1][2]) / 2.0)
        person_of = {k: rank + 1 for rank, k in enumerate(left_to_right)}

        # detectron2 hands back detections sorted by score, best first. Painting them in
        # reverse puts the most confident person on top where two people overlap.
        for k in reversed(range(len(decoded))):
            _i, (x0, y0, x1, y1), labels = decoded[k]
            where = labels > 0
            part_map[y0:y1, x0:x1][where] = labels[where]
            person_map[y0:y1, x0:x1][where] = person_of[k]

        if (sh, sw) != (h, w):
            part_map = cv2.resize(part_map, (w, h), interpolation=cv2.INTER_NEAREST)
            person_map = cv2.resize(person_map, (w, h), interpolation=cv2.INTER_NEAREST)

        inv = 1.0 / scale
        frame_boxes = [tuple(float(v) * inv for v in boxes[decoded[k][0]]) for k in left_to_right]
        return PartMap(part_map, person_map, frame_boxes, (w, h))
