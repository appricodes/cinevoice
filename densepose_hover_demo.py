"""
Standalone demo of the body explorer, so it can be tried on one picture without opening a
video in the player.

    python densepose_hover_demo.py <image-or-video> [--frame N] [--size 640]

It runs DensePose on the frame, lays the same invisible overlay over the picture that
body_explorer.py lays over the video, and reads out the body part under the pointer. With a
screen reader running, each body part is an object of its own there and the screen reader's
own mouse tracking announces it; the status line at the bottom always shows the same text,
so the mapping can be checked by eye as well.

Keys: O toggles the colour overlay, B toggles the person boxes, Esc quits.
"""
import argparse
import os
import sys

import cv2
from PySide6.QtCore import QObject, QPoint, QRect, Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget)

from body_explorer import PartOverlay
from densepose_parts import DensePoseParts, names_for_language, person_word_for_language


def load_frame(path, index=0):
    """One BGR frame from an image file, or from a video at the given frame index."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    image = cv2.imread(path)
    if image is not None:
        return image
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open {} as an image or a video".format(path))
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("could not read frame {} of {}".format(index, path))
    return frame


def _print_progress(done, total):
    print("\r  {:5.1f}%  {:,} / {:,} bytes".format(done * 100.0 / max(total, 1), done, total),
          end="", file=sys.stderr)


class AnalyzeWorker(QObject):
    """Runs the model off the GUI thread -- inference takes a second or two on CPU."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, frame, detect_size, language):
        super().__init__()
        self.frame = frame
        self.detect_size = detect_size
        self.language = language

    def run(self):
        try:
            engine = DensePoseParts(detect_size=self.detect_size)
            if not engine.available():
                print("Downloading the DensePose model, 265 MB, once...", file=sys.stderr)
                engine.ensure_model(progress=_print_progress)
                print("\nDone.", file=sys.stderr)
            part_map = engine.analyze(self.frame)
            regions = part_map.regions(names=names_for_language(self.language))
            self.done.emit((part_map, regions))
        except Exception as exc:  # noqa: BLE001 - the demo should show the error, not die
            self.failed.emit("{}: {}".format(type(exc).__name__, exc))


class FrameView(QLabel):
    """Shows the frame scaled to fit, and knows where inside itself the picture sits."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame = None
        self.part_map = None
        self.show_overlay = True
        self.show_boxes = True

    def set_result(self, part_map):
        self.part_map = part_map
        self.update()

    def picture_rect(self):
        """Where the picture is drawn inside this widget, in its own coordinates."""
        if self.frame is None:
            return QRect()
        h, w = self.frame.shape[:2]
        scale = min(self.width() / float(w), self.height() / float(h))
        shown_w, shown_h = round(w * scale), round(h * scale)
        return QRect((self.width() - shown_w) // 2, (self.height() - shown_h) // 2,
                     shown_w, shown_h)

    def paintEvent(self, event):
        if self.frame is None:
            super().paintEvent(event)
            return
        image = self.frame
        if self.part_map is not None and self.show_overlay:
            image = self.part_map.overlay(image)
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixmap = QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy())

        rect = self.picture_rect()
        painter = QPainter(self)
        painter.drawPixmap(QPoint(rect.x(), rect.y()),
                           pixmap.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))
        if self.part_map is not None and self.show_boxes:
            sx, sy = rect.width() / float(w), rect.height() / float(h)
            painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(255, 220, 0), 2))
            for i, (bx0, by0, bx1, by1) in enumerate(self.part_map.boxes, start=1):
                x, y = rect.x() + bx0 * sx, rect.y() + by0 * sy
                painter.drawRect(int(x), int(y), int((bx1 - bx0) * sx), int((by1 - by0) * sy))
                painter.drawText(int(x) + 4, int(y) + 18, str(i))
        painter.end()


class DemoWindow(QMainWindow):
    def __init__(self, frame, detect_size, language):
        super().__init__()
        self.setWindowTitle("DensePose — body parts")
        self.resize(900, 900)
        self.setMouseTracking(True)

        self.view = FrameView()
        self.view.frame = frame
        self.view.setMouseTracking(True)
        self.overlay = None
        self._last_text = None

        self.status = QLabel("Running DensePose on this frame...")
        self.status.setStyleSheet("padding: 6px; font-size: 14px;")

        central = QWidget()
        central.setMouseTracking(True)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.status)
        self.setCentralWidget(central)

        self._thread = QThread(self)
        self._worker = AnalyzeWorker(frame, detect_size, language)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_done)
        self._worker.failed.connect(self._on_failed)
        self._thread.start()

    def _on_done(self, result):
        self._thread.quit()
        part_map, regions = result
        self.view.set_result(part_map)
        if not regions:
            self.status.setText("No people found in this frame.")
            return
        n_people = len({r.person for r in regions})
        self.overlay = PartOverlay(self, regions, part_map.size,
                                   person_word=person_word_for_language(self._worker.language),
                                   single_person=(n_people == 1))
        self._place_overlay()
        self.overlay.show()
        self.status.setText("{} found, {} areas. Move the mouse over the picture.".format(
            "1 person" if n_people == 1 else "{} people".format(n_people), len(regions)))

    def _on_failed(self, message):
        self._thread.quit()
        self.status.setText("Error: " + message)

    def _place_overlay(self):
        if self.overlay is None:
            return
        rect = self.view.picture_rect()
        self.overlay.setGeometry(QRect(self.view.mapToGlobal(rect.topLeft()), rect.size()))

    def mouseMoveEvent(self, event):
        # The overlay is invisible and does not take focus, so the demo reads the part under
        # the pointer the same way the player does when no screen reader is running.
        if self.overlay is not None:
            rect = self.view.picture_rect()
            p = self.view.mapFromGlobal(event.globalPosition().toPoint()) - rect.topLeft()
            if 0 <= p.x() < rect.width() and 0 <= p.y() < rect.height():
                text = self.overlay.text_at(p.x(), p.y()) or "—"
            else:
                text = "Outside the picture."
            if text != self._last_text:
                self._last_text = text
                self.status.setText(text)
        super().mouseMoveEvent(event)

    def resizeEvent(self, event):
        self._place_overlay()
        super().resizeEvent(event)

    def moveEvent(self, event):
        self._place_overlay()
        super().moveEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_O:
            self.view.show_overlay = not self.view.show_overlay
            self.view.update()
        elif event.key() == Qt.Key.Key_B:
            self.view.show_boxes = not self.view.show_boxes
            self.view.update()
        elif event.key() == Qt.Key.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="image or video file")
    parser.add_argument("--frame", type=int, default=0, help="frame index, for videos")
    parser.add_argument("--size", type=int, default=640,
                        help="longest side the model sees; smaller is faster, coarser")
    parser.add_argument("--language", default="English",
                        help="language of the part names (English or Persian)")
    args = parser.parse_args(argv)

    frame = load_frame(args.path, args.frame)
    app = QApplication(sys.argv[:1])
    window = DemoWindow(frame, args.size, args.language)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
