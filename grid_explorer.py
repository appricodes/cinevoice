"""
Picture exploration: a mouse click on the video (or Ctrl+Q) pauses playback, turns the window
full screen, and has the AI describe the current frame as a grid of cells (4x3, 8x6 or 12x9
columns by rows, switched with Ctrl+Shift+Q); once the answer is in, moving the mouse over the
picture reads out the cell under the pointer. Space, Escape or another click ends it and resumes playback.

With a screen reader running, each cell is an object of its own on an invisible window laid
over the picture, labelled with the cell's text, and the screen reader's own mouse tracking
reads it -- cutting off the previous cell, as it does whenever the mouse reaches a new object.
Without one, the app's own voice reads the cell under the pointer.
"""
import json
import threading
from PySide6.QtCore import QObject, QEvent, QRect, Qt, Signal, Slot
from PySide6.QtGui import QWindow, QAccessible, QPainter, QColor
from PySide6.QtWidgets import QApplication, QWidget, QLabel

import settings
from config import GRID_EXPLORE_SIZES, GRID_EXPLORE_DEFAULT, GRID_EXPLORE_IMAGE_DIM, XAI_MAX_IMAGE_DIM
from utils import clean_string, resize_to_max_dim, encode_frame

IDLE, PENDING, ACTIVE = range(3)
# The pointer is over the window but off the picture itself (letterbox bars, the slider...).
_OUTSIDE = "outside"
_EMPTY_CELL = "Nothing described here."


def _row_letter(r: int) -> str:
    return chr(ord("A") + r)


def _cell_label(r: int, c: int) -> str:
    """A1 for the top-left cell: row letter, then 1-based column number."""
    return f"{_row_letter(r)}{c + 1}"


def _draw_grid(picture, cols: int, rows: int):
    """
    A copy of an OpenCV BGR picture with the cell borders drawn on it and each cell labelled
    in its top-left corner. Asked only to imagine the grid, models place things in the wrong
    cells and describe their neighbours; with the borders in sight they keep to each cell.
    """
    import cv2
    img = picture.copy()
    h, w = img.shape[:2]
    thick = max(1, round(min(w, h) / 400))
    xs = [round(c * w / cols) for c in range(cols + 1)]
    ys = [round(r * h / rows) for r in range(rows + 1)]
    # Yellow on a black outline, to stay visible over both light and dark scenes.
    for x in xs[1:-1]:
        cv2.line(img, (x, 0), (x, h - 1), (0, 0, 0), thick + 2)
        cv2.line(img, (x, 0), (x, h - 1), (0, 255, 255), thick)
    for y in ys[1:-1]:
        cv2.line(img, (0, y), (w - 1, y), (0, 0, 0), thick + 2)
        cv2.line(img, (0, y), (w - 1, y), (0, 255, 255), thick)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.3, min(0.8, min(w / cols, h / rows) / 120))
    for r in range(rows):
        for c in range(cols):
            label = _cell_label(r, c)
            (tw, th), base = cv2.getTextSize(label, font, scale, 1)
            x, y = xs[c] + thick + 1, ys[r] + thick + 1
            cv2.rectangle(img, (x, y), (x + tw + 4, y + th + base + 4), (0, 0, 0), -1)
            cv2.putText(img, label, (x + 2, y + th + 2), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


class _CellOverlay(QWidget):
    """
    A frameless window over the displayed picture holding one labelled object per cell, so a
    screen reader's mouse tracking finds a separate object under the pointer in every cell. It
    never takes focus, so Space and Escape still reach the main window.
    """

    def __init__(self, parent: QWidget, cells: list):
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setAccessibleName("Picture")
        self._rows, self._cols = len(cells), len(cells[0])
        self._labels = []
        for r, row in enumerate(cells):
            for c, text in enumerate(row):
                label = QLabel(self)
                label.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                label.setAccessibleName(text or _EMPTY_CELL)
                self._labels.append((r, c, label))

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        for r, c, label in self._labels:
            x1, x2 = round(c * w / self._cols), round((c + 1) * w / self._cols)
            y1, y2 = round(r * h / self._rows), round((r + 1) * h / self._rows)
            label.setGeometry(x1, y1, x2 - x1, y2 - y1)
        super().resizeEvent(event)

    def paintEvent(self, event):
        # Alpha 1 of 255: invisible, but not fully transparent -- Windows passes the mouse
        # straight through the fully transparent parts of a translucent window.
        QPainter(self).fillRect(self.rect(), QColor(0, 0, 0, 1))


class GridExplorer(QObject):
    exploreReady = Signal(int, object)  # request id, (cells, (frame_w, frame_h)) or an error string

    def __init__(self, main_window):
        super().__init__(main_window)
        self.mw = main_window
        self.state = IDLE
        self._cells = None
        self._frame_size = None
        self._last_cell = None
        self._overlay = None
        # The window goes full screen while exploring, so the picture is as large as it can be
        # under the mouse; this is the state to put back afterwards (normal or maximized).
        self._prev_window_state = None
        # Bumped on every start and stop, so an answer that arrives after the user already
        # gave up on it (or started over) is recognised as stale and dropped.
        self._request_id = 0
        # Index into GRID_EXPLORE_SIZES (Ctrl+Shift+Q), remembered between runs.
        self._size_index = self.mw.q_settings.value("grid_explore_size", GRID_EXPLORE_DEFAULT, type=int)
        if not 0 <= self._size_index < len(GRID_EXPLORE_SIZES):
            self._size_index = GRID_EXPLORE_DEFAULT
        self.exploreReady.connect(self._on_ready)
        # Application-wide rather than on the video widget: QVideoWidget draws through an
        # embedded native window, whose mouse events never reach the widget itself.
        QApplication.instance().installEventFilter(self)

    def is_engaged(self) -> bool:
        return self.state != IDLE

    def cycle_size(self):
        """Ctrl+Shift+Q: the next grid size, used from the next picture explored."""
        self._size_index = (self._size_index + 1) % len(GRID_EXPLORE_SIZES)
        self.mw.q_settings.setValue("grid_explore_size", self._size_index)
        cols, rows = GRID_EXPLORE_SIZES[self._size_index]
        later = " It applies from the next picture." if self.is_engaged() else ""
        self.mw.speak(f"Picture grid {cols} by {rows}.{later}")

    def start(self):
        if self.state == PENDING:
            self.mw.speak("Still analysing the picture.")
            return
        if self.state == ACTIVE:
            self.mw.speak("Already exploring. Click, or press Space or Escape, to resume playback.")
            return
        if not self.mw.media_path:
            self.mw.speak("No video currently open.")
            return
        self.mw.player.pause()
        self.state = PENDING
        self._request_id += 1
        self._last_cell = None
        self._prev_window_state = self.mw.windowState()
        self.mw.showFullScreen()
        self.mw.speak("Exploring the picture. Please wait.")
        self.mw.updateStatus.emit("Analysing the picture for exploration...")
        threading.Thread(target=self._worker,
                         args=(self._request_id, self.mw.media_path, self.mw.player.position(),
                               GRID_EXPLORE_SIZES[self._size_index]),
                         daemon=True).start()

    def stop(self, resume: bool = True):
        """Ends exploration (or abandons a pending request); resume=True restarts playback."""
        if self.state == PENDING and resume:
            self.mw.speak("Picture exploration cancelled.")
        self.state = IDLE
        self._request_id += 1
        self._cells = None
        self._remove_overlay()
        self._leave_full_screen()
        if resume:
            self.mw.player.play()

    def _leave_full_screen(self):
        if self._prev_window_state is None:
            return
        self.mw.setWindowState(self._prev_window_state)
        self._prev_window_state = None

    def _system_prompt(self, cols: int, rows: int) -> str:
        last_row = _row_letter(rows - 1)
        # The overall description is the one F2 ("Detail Desc") gives, at the same length and in
        # the same style, so arriving in the grid sounds like pressing F2 on the paused picture.
        f2 = settings.PROMPTS[1] if len(settings.PROMPTS) > 1 else {}
        f2_prompt = f2.get("prompt", "Describe this scene in detail.")
        max_words = f2.get("max_words", self.mw.current_max_words)
        return (f"You are helping a blind viewer explore a paused video picture by moving a pointer over it. "
                f"First, for 'description': {f2_prompt} Describe it as a professional cinematic audio describer "
                f"would, in a '{self.mw.current_mode}' style, in the immediate present tense, describing only what "
                f"is actually, physically present; under {max_words} words. "
                f"Then, for 'cells': the picture has a grid of labelled cells drawn over it, {cols} columns and "
                f"{rows} rows of equal cells. "
                f"Each cell is labelled with a letter for its row, A to {last_row} from top to bottom, and a number "
                f"for its column, 1 to {cols} from left to right: A1 is the top-left corner, {last_row}{cols} the "
                f"bottom-right. The lines and labels only mark the cells; they are not part of the scene. "
                f"For every cell, look at that cell's own area and say only what lies inside its borders -- never "
                f"what is in neighbouring cells, and no directions to other things. Do use the whole picture to "
                f"recognise what each part belongs to: a green stem whose flower is in the cell above is 'the stem "
                f"of the red flower', not just 'a green line'. Name the people, objects and "
                f"parts of them that fall within it as precisely as you can; if only part of something is inside, "
                f"name that part (for example 'the man's right shoulder' or 'the left edge of a red sofa'), with "
                f"colors and any visible text. If the cell holds only background, say just what it is (sky, wall, "
                f"floor...). Keep each cell under 15 words. Write everything in {self.mw.current_language}. "
                f"Never use the words 'cell', 'grid', 'row', 'column', 'image', 'frame', 'picture' or 'shows', and "
                f"never mention the labels or the lines. Reply with only a JSON object of this form, with an entry "
                f"for every one of the {cols * rows} labels: "
                f"{{\"description\": \"...\", \"cells\": {{\"A1\": \"...\", \"A2\": \"...\", ...}}}}.")

    def _worker(self, request_id: int, media_path: str, timestamp_ms: int, size: tuple):
        cols, rows = size
        import cv2
        try:
            cap = cv2.VideoCapture(media_path)
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok or frame is None:
                self.exploreReady.emit(request_id, "Error: Could not capture the current picture.")
                return
            frame_h, frame_w = frame.shape[:2]

            vm = self.mw.vision_manager
            max_dim = XAI_MAX_IMAGE_DIM if vm._is_xai_model() else GRID_EXPLORE_IMAGE_DIM
            # One image only: the whole picture with the grid drawn over it, so every cell is
            # read in the context of everything around it.
            gridded = _draw_grid(resize_to_max_dim(frame, max_dim), cols, rows)
            system_prompt = vm._with_custom_instruction(vm._with_cast(self._system_prompt(cols, rows)))

            response = self.mw.api_client.send(
                model_dict=self.mw.current_model_dict,
                frames_b64=[encode_frame(gridded, quality=90)],
                system_prompt=system_prompt,
                user_text_blocks=["Describe the scene, then each labelled cell, as JSON."],
                expect_json=True,
                language=self.mw.current_language
            )
            if not isinstance(response, dict) or "choices" not in response:
                error = response.get("error") if isinstance(response, dict) else None
                self.exploreReady.emit(request_id, f"Error: {error or 'Failed to analyse the picture.'}")
                return
            parsed = self._parse_reply(response["choices"][0]["message"]["content"] or "", cols, rows)
            if parsed is None:
                self.exploreReady.emit(request_id, "Error: The model's reply could not be read as a picture map.")
                return
            description, cells = parsed
            self.exploreReady.emit(request_id, (description, cells, (frame_w, frame_h)))
        except Exception as e:
            self.exploreReady.emit(request_id, f"Error exploring the picture: {e}")

    @staticmethod
    def _parse_reply(text: str, cols: int, rows: int):
        """
        (description, cells) from the reply -- cells as `rows` lists of `cols` strings, missing
        ones empty; the description "" if the model left it out -- or None if unreadable.
        """
        # Parsed from the outermost braces, in case the model wraps the JSON in prose or a code fence.
        try:
            data = json.loads(text[text.find("{"):text.rfind("}") + 1])
            reply = data.get("cells")
        except (ValueError, AttributeError):
            return None
        if not isinstance(reply, dict) or not reply:
            return None
        # Labels matched case- and space-insensitively ("a1", "A 1").
        reply = {str(k).replace(" ", "").upper(): v for k, v in reply.items()}
        cells = [[clean_string(str(reply.get(_cell_label(r, c)) or "")) for c in range(cols)] for r in range(rows)]
        return clean_string(str(data.get("description") or "")), cells

    @Slot(int, object)
    def _on_ready(self, request_id: int, result):
        if request_id != self._request_id or self.state != PENDING:
            return
        if isinstance(result, str):
            self.state = IDLE
            self._leave_full_screen()
            self.mw.updateStatus.emit(result)
            self.mw.speak(result)
            return
        description, self._cells, self._frame_size = result
        self.state = ACTIVE
        self._overlay = _CellOverlay(self.mw, self._cells)
        self._place_overlay()
        self._overlay.show()
        self.mw.updateStatus.emit("Picture ready to explore.")
        # The overall description first, as F2 would give it; moving the mouse cuts it off.
        self.mw.speak(f"{description or 'Picture ready.'} Move the mouse over the picture to hear each part. "
                      "Click, or press Space or Escape, to resume playback.")

    def _remove_overlay(self):
        if self._overlay is not None:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None

    def _picture_rect(self) -> QRect:
        """Where the picture is drawn inside the video widget, in the widget's coordinates."""
        vw = self.mw.video_widget
        frame_w, frame_h = self._frame_size
        # Scaled to fit the widget with its aspect ratio kept, centred between black bars.
        scale = min(vw.width() / frame_w, vw.height() / frame_h)
        shown_w, shown_h = round(frame_w * scale), round(frame_h * scale)
        return QRect((vw.width() - shown_w) // 2, (vw.height() - shown_h) // 2, shown_w, shown_h)

    def _place_overlay(self):
        rect = self._picture_rect()
        self._overlay.setGeometry(QRect(self.mw.video_widget.mapToGlobal(rect.topLeft()), rect.size()))

    def _over_video(self, window, global_pos) -> bool:
        """Whether a mouse event delivered to `window` is over the video widget of this player."""
        if QApplication.activeModalWidget() is not None:
            return False
        # The main window, a native window embedded in it (the video's own), or the cell overlay.
        ours = [self.mw.windowHandle()]
        if self._overlay is not None:
            ours.append(self._overlay.windowHandle())
        while window is not None and window not in ours:
            window = window.parent()
        if window is None:
            return False
        vw = self.mw.video_widget
        return vw.isVisible() and vw.rect().contains(vw.mapFromGlobal(global_pos))

    def _cell_at(self, window, global_pos):
        """(row, col) under the pointer, or _OUTSIDE if it is off the displayed picture."""
        if not self._over_video(window, global_pos):
            return _OUTSIDE
        rect = self._picture_rect()
        p = self.mw.video_widget.mapFromGlobal(global_pos) - rect.topLeft()
        if not (0 <= p.x() < rect.width() and 0 <= p.y() < rect.height()):
            return _OUTSIDE
        # The grid of the picture on screen, which a Ctrl+Shift+Q since has not changed.
        rows, cols = len(self._cells), len(self._cells[0])
        return min(rows - 1, p.y() * rows // rect.height()), min(cols - 1, p.x() * cols // rect.width())

    def eventFilter(self, obj, event):
        t = event.type()
        # The picture moved or changed size under the overlay (Alt+Enter, or the window moved).
        if (self._overlay is not None and t in (QEvent.Type.Resize, QEvent.Type.Move)
                and (obj is self.mw or obj is self.mw.video_widget)):
            self._place_overlay()
            return False
        # Only window-level events: each mouse event reaches exactly one QWindow (the main
        # window's, the video's embedded one, or the overlay's), so nothing is handled twice.
        # Never consumed.
        if t not in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseMove) or not isinstance(obj, QWindow):
            return False
        global_pos = event.globalPosition().toPoint()
        if t == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton and self._over_video(obj, global_pos):
                # A click toggles: starts exploring, or ends it like Space/Escape. While the
                # body explorer has the picture, the click belongs to it, not to the grid.
                body = getattr(self.mw, "body_explorer", None)
                if self.state != IDLE:
                    self.stop()
                elif body is None or not body.is_engaged():
                    self.start()
        elif self.state == ACTIVE and not QAccessible.isActive():
            # No screen reader to track the mouse over the overlay: read the cell ourselves.
            # QTextToSpeech.say() cuts off whatever it was still saying.
            cell = self._cell_at(obj, global_pos)
            if cell != self._last_cell:
                self._last_cell = cell
                if cell == _OUTSIDE:
                    self.mw.speak("Outside the picture.")
                else:
                    self.mw.speak(self._cells[cell[0]][cell[1]] or _EMPTY_CELL)
        return False
