"""
Body exploration: DensePose splits the paused picture into the body parts of everyone in
it, and moving the mouse over a person reads out which part is under the pointer and whose
it is -- "Person 2, left forearm".

This is the picture-grid feature of grid_explorer.py with the grid thrown away. Instead of
equal cells described by the AI, the areas here are the real shapes of the body parts, cut
by a local model: an arm is one area shaped like that arm, so the pointer crossing from the
sleeve onto the hand is heard as crossing onto the hand.

With a screen reader running, each area is an object of its own on an invisible window laid
over the picture, labelled with its part name and masked to that part's actual outline, so
the screen reader's own mouse tracking reads the part the pointer is really on. Without one,
the app's own voice reads it.

Everything here is offline: the model file lives under models/densepose/ and nothing is
sent anywhere.
"""
import importlib.util
import threading

import numpy as np
from PySide6.QtCore import QEvent, QObject, QRect, QSize, Qt, Signal, Slot
from PySide6.QtGui import QAccessible, QBitmap, QColor, QImage, QPainter, QRegion, QWindow
from PySide6.QtWidgets import QApplication, QLabel, QWidget

IDLE, PENDING, ACTIVE = range(3)
# The pointer is over the window but off the picture itself (letterbox bars, the slider...).
_OUTSIDE = "outside"
_NOBODY = "Nobody here."
# Said in place of "Person 1" when only one person was found: there is nothing to tell apart.
_SINGLE_PERSON_WORD = ""


def _qregion_from_mask(mask):
    """
    A QRegion covering exactly the true pixels of a bool mask.

    Built from raw 1-bit data rather than by thresholding a grey image, because a mono
    conversion picks its own polarity and would just as happily hand back the inverse of
    the body part. QBitmap.fromData takes (width + 7) // 8 bytes per row -- no padding --
    and QRegion covers the bits that are 1.

    Going through a Format_Mono QImage and QBitmap.fromImage looks equivalent and is not:
    such an image has no colour table, and converting it crashes the process.
    """
    height, width = mask.shape
    if not height or not width:
        return QRegion()
    packed = np.packbits(np.ascontiguousarray(mask, dtype=np.uint8), axis=1, bitorder="little")
    bitmap = QBitmap.fromData(QSize(width, height), packed.tobytes(),
                              QImage.Format.Format_MonoLSB)
    return QRegion(bitmap)


class PartOverlay(QWidget):
    """
    A frameless window over the displayed picture holding one labelled object per body
    part, each masked to that part's outline, so a screen reader's mouse tracking finds a
    separate object under the pointer on every part. It never takes focus, so Space and
    Escape still reach the main window.
    """

    def __init__(self, parent, regions, frame_size, person_word="Person", single_person=False):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName("Picture")
        self._frame_w, self._frame_h = frame_size
        self._regions = regions
        self._labels = []
        # Largest first, so the small parts are created last and sit on top. The masks do
        # not overlap, but a child added later also wins any rounding along a shared edge.
        for region in regions:
            label = QLabel(self)
            label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            text = region.name if single_person else region.label(person_word)
            label.setAccessibleName(text)
            self._labels.append((region, label))

    def text_at(self, x, y):
        """The announcement for a point in overlay coordinates, or None if off every part."""
        if self.width() <= 0 or self.height() <= 0:
            return None
        fx = int(x * self._frame_w / float(self.width()))
        fy = int(y * self._frame_h / float(self.height()))
        if not (0 <= fx < self._frame_w and 0 <= fy < self._frame_h):
            return None
        for region, label in self._labels:
            x0, y0, x1, y1 = region.bbox
            if x0 <= fx < x1 and y0 <= fy < y1 and region.mask[fy, fx]:
                return label.accessibleName()
        return None

    def resizeEvent(self, event):
        import cv2
        width, height = self.width(), self.height()
        if width > 0 and height > 0:
            for region, label in self._labels:
                mask = cv2.resize(region.mask.astype(np.uint8), (width, height),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
                where = np.argwhere(mask)
                if not len(where):
                    label.setGeometry(0, 0, 0, 0)
                    continue
                (y0, x0), (y1, x1) = where.min(0), where.max(0) + 1
                label.setGeometry(int(x0), int(y0), int(x1 - x0), int(y1 - y0))
                label.setMask(_qregion_from_mask(mask[y0:y1, x0:x1]))
        super().resizeEvent(event)

    def paintEvent(self, event):
        # Alpha 1 of 255: invisible, but not fully transparent -- Windows passes the mouse
        # straight through the fully transparent parts of a translucent window.
        QPainter(self).fillRect(self.rect(), QColor(0, 0, 0, 1))


class BodyExplorer(QObject):
    """Mirrors GridExplorer, with a local DensePose pass in place of the AI grid call."""

    exploreReady = Signal(int, object)  # request id, the finished areas or an error string
    announce = Signal(str, bool)        # text, and whether it also goes to the status line

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = IDLE
        self._frame_size = None
        self._overlay = None
        self._last_text = None
        self._prev_window_state = None
        self._request_id = 0
        # densepose_parts pulls in torch, a second or so of import time the player should
        # not pay on every start, so it is loaded the first time the feature is used.
        self._engine = None
        self.exploreReady.connect(self._on_ready)
        self.announce.connect(self._on_announce)
        # Application-wide rather than on the video widget: QVideoWidget draws through an
        # embedded native window, whose mouse events never reach the widget itself.
        QApplication.instance().installEventFilter(self)

    def is_engaged(self):
        return self.state != IDLE

    def _get_engine(self):
        if self._engine is None:
            from densepose_parts import DensePoseParts
            self._engine = DensePoseParts()
        return self._engine

    def model_installed(self):
        """Whether the model file is there. densepose_parts imports torch only lazily,
        so asking this does not drag torch in."""
        return self._get_engine().available()

    def start(self):
        if self.state == PENDING:
            self.mw.speak("Still finding the people in the picture.")
            return
        if self.state == ACTIVE:
            self.mw.speak("Already exploring. Click, or press Space or Escape, to resume playback.")
            return
        if not self.mw.media_path:
            self.mw.speak("No video currently open.")
            return
        grid = getattr(self.mw, "grid_explorer", None)
        if grid is not None and grid.is_engaged():
            # Both lay an overlay over the same picture and both go full screen; the one
            # already running keeps the picture.
            self.mw.speak("Picture exploration is running. End it first.")
            return
        if importlib.util.find_spec("torch") is None:
            # Checked before the thread starts, so the reason is spoken rather than
            # arriving as an ImportError in the middle of the worker.
            self.mw.speak("Body exploration needs PyTorch installed.")
            self.mw.updateStatus.emit(
                "PyTorch is not installed: pip install torch torchvision "
                "--index-url https://download.pytorch.org/whl/cpu")
            return
        self.mw.player.pause()
        self.state = PENDING
        self._request_id += 1
        self._last_text = None
        self._prev_window_state = self.mw.windowState()
        self.mw.showFullScreen()
        if self.model_installed():
            self.mw.speak("Finding the people in the picture. Please wait.")
            self.mw.updateStatus.emit("Finding body parts...")
        else:
            # Said before the thread starts, so the wait is never unexplained. Escape or
            # Space stops it, and what has already arrived is kept for the next attempt.
            from densepose_parts import MODEL_BYTES
            done = self._get_engine().partial_bytes()
            left = max(1, (MODEL_BYTES - done) // (1024 * 1024))
            unit = "megabyte" if left == 1 else "megabytes"
            if done:
                self.mw.speak("Carrying on with the body model download, {} {} left. "
                              "Press Escape to stop.".format(left, unit))
            else:
                self.mw.speak("The body model is not here yet. Downloading it now, {} {}; "
                              "this happens once. Press Escape to stop.".format(left, unit))
            self.mw.updateStatus.emit("Downloading the body model ({} {} left)...".format(left, unit))
        threading.Thread(target=self._worker,
                         args=(self._request_id, self.mw.media_path, self.mw.player.position()),
                         daemon=True).start()

    def stop(self, resume=True):
        """Ends exploration (or abandons a pending request); resume=True restarts playback."""
        if self.state == PENDING and resume:
            if self.model_installed():
                self.mw.speak("Body exploration cancelled.")
            else:
                self.mw.speak("Download stopped. What has arrived is kept for next time.")
        self.state = IDLE
        self._request_id += 1
        self._remove_overlay()
        self._leave_full_screen()
        if resume:
            self.mw.player.play()

    def _leave_full_screen(self):
        if self._prev_window_state is None:
            return
        self.mw.setWindowState(self._prev_window_state)
        self._prev_window_state = None

    def _download_callbacks(self, request_id):
        """(progress, cancelled) for download_model, wired to speech and to stop()."""
        # Only on a tenth of the file, so the download is audibly moving without talking
        # over itself; a resumed one starts partway up and says so.
        said = [-1]

        def progress(done, total):
            tenth = int(done * 10 / max(total, 1))
            if tenth > said[0]:
                said[0] = tenth
                if 0 < tenth < 10:
                    self.announce.emit("Body model {} percent.".format(tenth * 10), False)

        def cancelled():
            # stop() bumps the request id; that is what ends the download.
            return request_id != self._request_id

        return progress, cancelled

    def _worker(self, request_id, media_path, timestamp_ms):
        import cv2
        from densepose_parts import DownloadCancelled
        try:
            if not self.model_installed():
                progress, cancelled = self._download_callbacks(request_id)
                try:
                    self._get_engine().ensure_model(progress=progress, cancelled=cancelled)
                except DownloadCancelled:
                    return
                except Exception as e:
                    self.exploreReady.emit(
                        request_id, "Error: Could not download the body model: {}".format(e))
                    return
                if request_id != self._request_id:
                    return
                self.announce.emit("Body model ready. Finding the people in the picture.", True)
            cap = cv2.VideoCapture(media_path)
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok or frame is None:
                self.exploreReady.emit(request_id, "Error: Could not capture the current picture.")
                return
            from densepose_parts import names_for_language, person_word_for_language
            part_map = self._get_engine().analyze(frame)
            language = getattr(self.mw, "current_language", "English")
            regions = part_map.regions(names=names_for_language(language))
            person_word = person_word_for_language(language)
            # Count the people who actually have areas to explore, not every detection:
            # regions() drops anyone too small to put a pointer on.
            n_people = len({r.person for r in regions})
            frame_h, frame_w = frame.shape[:2]
            self.exploreReady.emit(request_id,
                                   (regions, n_people, person_word, (frame_w, frame_h)))
        except Exception as e:
            self.exploreReady.emit(request_id, "Error exploring the body: {}".format(e))

    @Slot(str, bool)
    def _on_announce(self, text, to_status):
        """Progress from the worker thread, spoken on the GUI thread."""
        if self.state != PENDING:
            return
        self.mw.speak(text)
        if to_status:
            self.mw.updateStatus.emit(text)

    @Slot(int, object)
    def _on_ready(self, request_id, result):
        if request_id != self._request_id or self.state != PENDING:
            return
        if isinstance(result, str):
            self.state = IDLE
            self._leave_full_screen()
            self.mw.updateStatus.emit(result)
            self.mw.speak(result)
            return
        regions, n_people, person_word, self._frame_size = result
        if not regions:
            self.state = IDLE
            self._leave_full_screen()
            self.mw.updateStatus.emit("No people found in this picture.")
            self.mw.speak("No people found in this picture.")
            return
        self.state = ACTIVE
        self._overlay = PartOverlay(self.mw, regions, self._frame_size,
                                    person_word=person_word, single_person=(n_people == 1))
        self._place_overlay()
        self._overlay.show()
        self.mw.updateStatus.emit("{} found. Move the mouse over the picture.".format(
            "One person" if n_people == 1 else "{} people".format(n_people)))
        self.mw.speak("{} in the picture. Move the mouse over them to hear each body part. "
                      "Click, or press Space or Escape, to resume playback.".format(
                          "One person" if n_people == 1 else "{} people".format(n_people)))

    def _remove_overlay(self):
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None

    def _picture_rect(self):
        """Where the picture is drawn inside the video widget, in the widget's coordinates."""
        vw = self.mw.video_widget
        frame_w, frame_h = self._frame_size
        scale = min(vw.width() / frame_w, vw.height() / frame_h)
        shown_w, shown_h = round(frame_w * scale), round(frame_h * scale)
        return QRect((vw.width() - shown_w) // 2, (vw.height() - shown_h) // 2, shown_w, shown_h)

    def _place_overlay(self):
        rect = self._picture_rect()
        self._overlay.setGeometry(QRect(self.mw.video_widget.mapToGlobal(rect.topLeft()), rect.size()))

    def _over_video(self, window, global_pos):
        """Whether a mouse event delivered to `window` is over the video widget of this player."""
        if QApplication.activeModalWidget() is not None:
            return False
        ours = [self.mw.windowHandle()]
        if self._overlay is not None:
            ours.append(self._overlay.windowHandle())
        while window is not None and window not in ours:
            window = window.parent()
        if window is None:
            return False
        vw = self.mw.video_widget
        return vw.isVisible() and vw.rect().contains(vw.mapFromGlobal(global_pos))

    def _text_at(self, window, global_pos):
        """What to say for the pointer position, or _OUTSIDE if it is off the picture."""
        if not self._over_video(window, global_pos):
            return _OUTSIDE
        rect = self._picture_rect()
        p = self.mw.video_widget.mapFromGlobal(global_pos) - rect.topLeft()
        if not (0 <= p.x() < rect.width() and 0 <= p.y() < rect.height()):
            return _OUTSIDE
        return self._overlay.text_at(p.x(), p.y()) or _NOBODY

    def eventFilter(self, obj, event):
        t = event.type()
        # The picture moved or changed size under the overlay (Alt+Enter, or the window moved).
        if (self._overlay is not None and t in (QEvent.Type.Resize, QEvent.Type.Move)
                and (obj is self.mw or obj is self.mw.video_widget)):
            self._place_overlay()
            return False
        if t not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove) or not isinstance(obj, QWindow):
            return False
        global_pos = event.globalPosition().toPoint()
        if t == QEvent.Type.MouseButtonPress:
            if (self.state == ACTIVE and event.button() == Qt.MouseButton.LeftButton
                    and self._over_video(obj, global_pos)):
                self.stop()
                # Consumed, so the same click does not then start the picture grid.
                return True
        elif self.state == ACTIVE and not QAccessible.isActive():
            # No screen reader to track the mouse over the overlay: read the part ourselves.
            # QTextToSpeech.say() cuts off whatever it was still saying.
            text = self._text_at(obj, global_pos)
            if text != self._last_text:
                self._last_text = text
                self.mw.speak("Outside the picture." if text == _OUTSIDE else text)
        return False
