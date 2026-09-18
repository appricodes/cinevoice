"""Dialog windows and small custom widgets shared across the app."""
from PySide6.QtWidgets import (
    QApplication, QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QPushButton, QDialogButtonBox, QLineEdit,
    QTextEdit, QSlider, QMessageBox, QFormLayout, QListWidget
)
import threading
from PySide6.QtCore import Qt, Signal
from PySide6 import QtGui
from config import LOCAL_MODELS


def announce_for_screen_reader(widget, text: str):
    """Pushes a screen reader announcement anchored to a widget without moving focus."""
    widget.setAccessibleName(text)
    announcement_event = getattr(QtGui, "QAccessibleAnnouncementEvent", None)
    if announcement_event and QtGui.QAccessible.isActive():
        try:
            ev = announcement_event(widget, text)
            QtGui.QAccessible.updateAccessibility(ev)
        except Exception:
            pass

class ClickableSlider(QSlider):
    """QSlider that jumps to the clicked position instead of Qt's default page-step behavior."""
    sliderClicked = Signal(int)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            val = self.style().sliderValueFromPosition(
                self.minimum(), self.maximum(), int(event.position().x()), self.width()
            )
            self.setValue(val)
            self.sliderClicked.emit(val)
            event.accept()
        super().mousePressEvent(event)

class CustomPromptPanel(QWidget):
    """
    Lets the user type a free-form question and pick how many frames to send with it.
    The class-level last_* fields remember the previous values in memory only, so the next
    time this panel opens (Ctrl+A) within the same run it's pre-filled with what was asked
    before -- nothing here is written to disk, so none of it survives closing the app.

    This is a plain child widget, embedded in the main window and hidden until needed
    (open_for_input), rather than a separate QDialog. A modal dialog opening and then closing
    changes the OS foreground window twice, and screen readers announce the newly-foregrounded
    window's full title on each change -- for the main window that title includes the open
    file's name, so a screen reader user had to sit through it, however long, before hearing
    anything else every time this closed. Showing/hiding a child widget never changes the
    foreground window, so that re-announcement doesn't happen; only what the app explicitly
    speaks (via MainWindow.speak) is heard.
    """
    last_prompt = ""
    last_frames_index = 0
    last_interval_index = 1
    last_words_index = 2
    last_frames_val = 1
    last_interval_val = 0.5
    last_words_val = 100

    # prompt_text, frames_count, frames_interval, max_words
    accepted = Signal(str, int, float, int)
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.prompt_label = QLabel("Enter your custom prompt:")
        self.prompt_edit = QLineEdit(self)

        self.frames_label = QLabel("Number of frames:")
        self.frames_combo = QComboBox(self)
        self.frames_combo.addItems(["1", "2", "3", "5", "10", "20", "50", "100"])

        self.interval_label = QLabel("Frame intervals (sec):")
        self.interval_combo = QComboBox(self)
        self.interval_combo.addItems(["0.25", "0.5", "1", "2", "5", "10"])

        self.words_label = QLabel("Max reply length (words):")
        self.words_combo = QComboBox(self)
        self.words_combo.addItems(["10", "50", "100", "250", "500", "1000", "2000", "4000"])

        self.ask_button = QPushButton("Ask", self)
        # QPushButton.autoDefault only defaults to True when its top-level parent is a
        # QDialog; ours is a plain QWidget, so without this Enter would do nothing while the
        # button has focus (Space would still work, since that's unconditional).
        self.ask_button.setAutoDefault(True)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.prompt_label)
        main_layout.addWidget(self.prompt_edit)

        frames_layout = QHBoxLayout()
        frames_layout.addWidget(self.frames_label)
        frames_layout.addWidget(self.frames_combo, 1)
        main_layout.addLayout(frames_layout)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(self.interval_label)
        interval_layout.addWidget(self.interval_combo, 1)
        main_layout.addLayout(interval_layout)

        words_layout = QHBoxLayout()
        words_layout.addWidget(self.words_label)
        words_layout.addWidget(self.words_combo, 1)
        main_layout.addLayout(words_layout)

        main_layout.addWidget(self.ask_button)
        self.setLayout(main_layout)

        self.ask_button.clicked.connect(self._on_accept)
        self.prompt_edit.returnPressed.connect(self._on_accept)

        # Fires regardless of which child widget currently holds focus, mirroring the
        # Escape-closes-the-dialog behavior QDialog used to give this for free.
        escape_shortcut = QtGui.QShortcut(QtGui.QKeySequence(Qt.Key.Key_Escape), self)
        escape_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        escape_shortcut.activated.connect(self._on_cancel)

        self.setVisible(False)

    @staticmethod
    def _set_index(combo, index):
        """Selects a restored index, clamped: a value saved by a different version of this
        panel could fall outside the current list of choices."""
        combo.setCurrentIndex(max(0, min(int(index), combo.count() - 1)))

    def open_for_input(self):
        """Shows the panel pre-filled with the last-used values, ready for typing."""
        self.prompt_edit.setText(CustomPromptPanel.last_prompt)
        self._set_index(self.frames_combo, CustomPromptPanel.last_frames_index)
        self._set_index(self.interval_combo, CustomPromptPanel.last_interval_index)
        self._set_index(self.words_combo, CustomPromptPanel.last_words_index)
        self.setVisible(True)
        # Select the pre-filled text from last time so typing replaces it immediately.
        self.prompt_edit.setFocus()
        self.prompt_edit.selectAll()

    def _on_accept(self):
        CustomPromptPanel.last_prompt = self.prompt_edit.text()
        CustomPromptPanel.last_frames_index = self.frames_combo.currentIndex()
        CustomPromptPanel.last_interval_index = self.interval_combo.currentIndex()
        CustomPromptPanel.last_words_index = self.words_combo.currentIndex()
        CustomPromptPanel.last_frames_val = int(self.frames_combo.currentText())
        CustomPromptPanel.last_interval_val = float(self.interval_combo.currentText())
        CustomPromptPanel.last_words_val = int(self.words_combo.currentText())
        self.setVisible(False)
        self.accepted.emit(CustomPromptPanel.last_prompt, CustomPromptPanel.last_frames_val,
                            CustomPromptPanel.last_interval_val, CustomPromptPanel.last_words_val)

    def _on_cancel(self):
        if not self.isVisible():
            return
        self.setVisible(False)
        self.cancelled.emit()

class ApiKeyDialog(QDialog):
    """One password-masked field per provider; a saved key shows as "********" instead of its value."""
    def __init__(self, current_keys_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage API Keys")
        self.setMinimumWidth(400)
        self.label = QLabel("Enter API keys. Leave field blank to delete a key. Hidden keys are safe.")
        
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        
        form = QFormLayout()
        self.edits = {}
        
        from auth import get_unique_providers
        providers = get_unique_providers()
        
        for provider_id, provider_name in providers.items():
            edit = QLineEdit(self)
            edit.setEchoMode(QLineEdit.EchoMode.Password)
            if current_keys_dict.get(provider_id):
                edit.setText("********")
            self.edits[provider_id] = edit
            form.addRow(f"{provider_name} API Key:", edit)

        layout.addLayout(form)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Save")
        layout.addWidget(self.buttons)
        self.setLayout(layout)
        
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        
    def get_keys(self):
        return {pid: edit.text().strip() for pid, edit in self.edits.items()}

class ModelMenuDialog(QDialog):
    """List of every model (online and downloaded offline); flags any online model missing its API key."""
    def __init__(self, models_list, current_model_id, api_keys_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select AI Model (Press Escape to choose)")
        self.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        
        current_row = 0
        for i, m in enumerate(models_list):
            is_local = m.get("provider_id") == "local"
            has_key = True if is_local else bool(api_keys_dict.get(m["provider_id"]))
            prefix = "[x] " if m["model_id"] == current_model_id else "[ ] "
            # An offline entry's name already says everything that matters (size, quality);
            # repeating the provider and a no-API-key note on every line only lengthens what
            # a screen reader has to read out before reaching the next model.
            if is_local:
                text = f"{prefix}{m['model_name']}"
            else:
                suffix = "" if has_key else " [Add API Key]"
                text = f"{prefix}{m['model_name']} ({m['provider_name']}){suffix}"
            
            self.list_widget.addItem(text)
            if m["model_id"] == current_model_id:
                current_row = i
                
        layout.addWidget(self.list_widget)
        self.list_widget.setCurrentRow(current_row)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self.accept()
        else:
            super().keyPressEvent(event)

    def get_selected_index(self):
        return self.list_widget.currentRow()

class CommandMenuDialog(QDialog):
    """Browsable list of commands (Ctrl+Down Arrow) for users who don't remember every shortcut."""
    def __init__(self, commands_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Commands Menu (Press Escape to Run)")
        self.setMinimumSize(400, 500)
        
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget(self)
        
        for cmd in commands_dict.keys():
            self.list_widget.addItem(cmd)
            
        layout.addWidget(self.list_widget)
        self.list_widget.setCurrentRow(0)
        self.setLayout(layout)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self.accept()
        else:
            super().keyPressEvent(event)

    def get_selected(self):
        item = self.list_widget.currentItem()
        return item.text() if item else None

class LastOutputDialog(QDialog):
    """Shows the last thing the app spoke, for re-reading or copying (Ctrl+R)."""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Last Response")
        self.setMinimumSize(500, 400)
        layout = QVBoxLayout(self)
        
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(text)
        layout.addWidget(self.text_edit)
        
        btn_layout = QHBoxLayout()
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.close_btn = QPushButton("Close")
        btn_layout.addStretch()
        btn_layout.addWidget(self.copy_btn)
        btn_layout.addWidget(self.close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
        self.copy_btn.clicked.connect(self.copy_text)
        self.close_btn.clicked.connect(self.accept)
        
    def copy_text(self):
        QApplication.clipboard().setText(self.text_edit.toPlainText())

class ActivationDialog(QDialog):
    """
    Base dialog that keeps itself frontmost and focused. If it is opened while
    the application is in the background, or the user alt-tabs away and returns,
    it re-activates itself whenever the application becomes active again.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        app = QApplication.instance()
        if app is not None:
            app.applicationStateChanged.connect(self._on_app_state)

    def _on_app_state(self, state):
        if state != Qt.ApplicationState.ApplicationActive or not self.isVisible():
            return
        modal = QApplication.activeModalWidget()
        if modal is not None and modal is not self:
            return  # a child modal (e.g. the success dialog) is on top; leave it focused
        self.raise_()
        self.activateWindow()

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        self.activateWindow()

    def done(self, result):
        app = QApplication.instance()
        if app is not None:
            try:
                app.applicationStateChanged.disconnect(self._on_app_state)
            except Exception:
                pass
        super().done(result)


class DownloadSuccessDialog(ActivationDialog):
    """
    Confirmation shown after a model finishes downloading. The message lives in a
    read-only text area so a screen reader user can Tab into it and review the
    full text; pressing Enter or OK closes it (and the download dialog behind it).
    """
    def __init__(self, message, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Complete")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setPlainText(message)
        self.text.setTabChangesFocus(True)
        self.text.setAccessibleName("Download result")
        layout.addWidget(self.text)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setDefault(True)
        self.button_box.accepted.connect(self.accept)
        ok_button.setFocus()
        announce_for_screen_reader(self.text, message)


class DeleteConfirmDialog(ActivationDialog):
    """
    Confirmation shown before permanently deleting a downloaded offline model.
    The explanation lives in a read-only text area so a screen reader user can
    Tab into it and review it in full. Cancel is the default and initially
    focused button so an accidental Enter press cannot trigger a deletion.
    """
    def __init__(self, model_dict, is_current_model, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Delete {model_dict['model_name']}?")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        message = (
            f"You are about to permanently delete \"{model_dict['model_name']}\" from your computer. "
            f"This removes all of its files from disk, freeing about {model_dict['size_gb']} gigabytes of space. "
            "This cannot be undone. If you want to use this model again later, you will need to download it again."
        )
        if is_current_model:
            message += (" This is your currently active AI model. After deletion the player will "
                       "switch back to the default online model.")
        message += " Activate Delete Model to confirm, or Cancel to keep it."

        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setPlainText(message)
        self.text.setTabChangesFocus(True)
        self.text.setAccessibleName("Delete confirmation details")
        layout.addWidget(self.text)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setText("Delete Model")
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setDefault(True)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        cancel_button.setFocus()
        announce_for_screen_reader(self.text, message)


class ExistingDescriptionDialog(ActivationDialog):
    """
    Shown when batch-generating narration for the whole video finds block descriptions
    already saved from an earlier run. The explanation lives in a read-only text area so
    a screen reader user can Tab into it and review it in full. Cancel is the default and
    initially focused button so an accidental Enter press can't wipe out existing work.
    """
    def __init__(self, block_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Descriptions Already Exist")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        message = (
            f"This video already has {block_count} saved 30-second block description"
            f"{'s' if block_count != 1 else ''}. "
            "Regenerate deletes them and describes the whole video again from scratch. "
            "Use Existing keeps them and turns on continuous 30-second narration, the same as Ctrl+D, "
            "describing any remaining blocks as you reach them. "
            "Cancel leaves everything as it is and starts nothing."
        )
        self.text = QTextEdit(self)
        self.text.setReadOnly(True)
        self.text.setPlainText(message)
        self.text.setTabChangesFocus(True)
        self.text.setAccessibleName("Existing description choice details")
        layout.addWidget(self.text)

        self.choice = "cancel"
        self.button_box = QDialogButtonBox()
        regenerate_button = self.button_box.addButton("Regenerate", QDialogButtonBox.ButtonRole.DestructiveRole)
        use_existing_button = self.button_box.addButton("Use Existing", QDialogButtonBox.ButtonRole.AcceptRole)
        cancel_button = self.button_box.addButton("Cancel", QDialogButtonBox.ButtonRole.RejectRole)
        cancel_button.setDefault(True)
        layout.addWidget(self.button_box)
        self.setLayout(layout)

        regenerate_button.clicked.connect(lambda: self._pick("regenerate"))
        use_existing_button.clicked.connect(lambda: self._pick("use_existing"))
        self.button_box.rejected.connect(self.reject)
        cancel_button.setFocus()
        announce_for_screen_reader(self.text, message)

    def _pick(self, choice: str):
        self.choice = choice
        self.accept()


class DownloadProgressDialog(ActivationDialog):
    """
    Modal dialog that downloads an offline model in a background thread. The
    percentage is shown in the window title and refreshed on every one percent
    step. A Cancel Download button aborts (resumable next time); on success it
    shows a confirmation dialog and returns Accepted with .success set.
    """
    progressChanged = Signal(int)
    downloadFinished = Signal(bool, str)

    def __init__(self, model_dict, parent=None):
        super().__init__(parent)
        self.model_dict = model_dict
        self.success = False
        self._finished = False
        self._cancel_event = threading.Event()
        self._last_percent = -1

        self.setWindowTitle(f"Downloading {model_dict['model_name']} 0%")
        self.setMinimumWidth(440)
        layout = QVBoxLayout(self)

        self.status_label = QLabel(
            f"Downloading {model_dict['model_name']}, about {model_dict['size_gb']} gigabytes. "
            "The percentage is shown in this window's title.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.cancel_button = QPushButton("Cancel Download", self)
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.setLayout(layout)
        self.cancel_button.setFocus()

        self.progressChanged.connect(self.on_progress)
        self.downloadFinished.connect(self.on_finished)

        def worker():
            import local_vlm
            ok, msg = local_vlm.download_model(
                model_dict,
                progress_cb=self.progressChanged.emit,
                cancel_event=self._cancel_event
            )
            self.downloadFinished.emit(ok, msg)

        threading.Thread(target=worker, daemon=True).start()

    def on_progress(self, percent: int):
        if percent == self._last_percent:
            return
        self._last_percent = percent
        self.setWindowTitle(f"Downloading {self.model_dict['model_name']} {percent}%")
        self.status_label.setText(f"Downloading... {percent} percent")
        announce_for_screen_reader(self.status_label, f"{percent} percent")

    def on_finished(self, ok: bool, message: str):
        self._finished = True
        if ok:
            self.success = True
            self.setWindowTitle(f"Downloaded {self.model_dict['model_name']} 100%")
            msg = (f"Download complete. {self.model_dict['model_name']} is now your current AI model. "
                   "Reminder: offline models run on your own computer. They are slow, and they are not "
                   "suitable for describing videos with continuous narration. Use them for single frame descriptions.")
            DownloadSuccessDialog(msg, self).exec()
            self.accept()
        else:
            self.status_label.setText(message)
            if not self._cancel_event.is_set():
                QMessageBox.warning(self, "Download Failed", message)
            self.reject()

    def reject(self):
        # First press cancels the download; the dialog closes once the worker confirms.
        if not self._finished:
            self._cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancelling download...")
            announce_for_screen_reader(self.status_label, "Cancelling download.")
            return
        super().reject()


class SettingsDialog(QDialog):
    """
    Language, TTS voice/speed, API keys, and offline model management.
    Two attributes let the caller react after the dialog closes:
    .pending_download_model is set (and the dialog auto-accepts) when the user chose
    to download a model, so the caller can run that download in its own dialog.
    .model_deleted_while_active is set if the user deleted the model that was
    currently in use, so the caller can fall back to a default online model.
    """
    def __init__(self, languages_list, current_language, tts_voices_list, current_voice, current_speed, manage_keys_callback, current_model_id=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(400)
        main_layout = QVBoxLayout(self)

        # --- Offline model download/delete section (shown first) ---
        offline_layout = QHBoxLayout()
        self.offline_label = QLabel("Offline AI Model:")
        self.offline_combo = QComboBox(self)
        self.offline_label.setBuddy(self.offline_combo)
        offline_layout.addWidget(self.offline_label)
        offline_layout.addWidget(self.offline_combo, 1)
        main_layout.addLayout(offline_layout)

        self.action_button = QPushButton("Download Selected Offline Model", self)
        main_layout.addWidget(self.action_button)

        self.download_status_label = QLabel("Offline models run on this computer without internet or API keys. They are slower than online models.")
        self.download_status_label.setWordWrap(True)
        main_layout.addWidget(self.download_status_label)

        self.current_model_id = current_model_id
        self.pending_download_model = None
        self.model_deleted_while_active = None
        self._populate_offline_combo()
        self.offline_combo.currentIndexChanged.connect(self._update_action_button)
        self.action_button.clicked.connect(self.on_action_button_clicked)
        self._update_action_button()
        main_layout.addStretch()

        lang_layout = QHBoxLayout()
        self.lang_label = QLabel("Spoken Language:")
        self.lang_combo = QComboBox(self)
        self.lang_combo.addItems(languages_list)
        self.lang_combo.setCurrentText(current_language)
        self.lang_label.setBuddy(self.lang_combo)
        lang_layout.addWidget(self.lang_label)
        lang_layout.addWidget(self.lang_combo, 1)
        main_layout.addLayout(lang_layout)
        
        voice_layout = QHBoxLayout()
        self.voice_label = QLabel("System TTS Voice:")
        self.voice_combo = QComboBox(self)
        self.voice_combo.addItems(tts_voices_list)
        self.voice_combo.setCurrentText(current_voice)
        self.voice_label.setBuddy(self.voice_combo)
        voice_layout.addWidget(self.voice_label)
        voice_layout.addWidget(self.voice_combo, 1)
        main_layout.addLayout(voice_layout)
        
        speed_layout = QHBoxLayout()
        self.speed_label = QLabel("TTS Speed (Percent):")
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(-100, 100)
        self.speed_slider.setValue(int(current_speed * 100))
        
        self.speed_label.setBuddy(self.speed_slider)
        self.speed_slider.setAccessibleName("TTS Speech Speed Percentage")
        
        self.speed_val_label = QLabel(f"{int(current_speed * 100)}%")
        self.speed_slider.valueChanged.connect(lambda v: self.speed_val_label.setText(f"{v}%"))
        
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_slider, 1)
        speed_layout.addWidget(self.speed_val_label)
        main_layout.addLayout(speed_layout)
        
        main_layout.addStretch()

        self.manage_keys_button = QPushButton("Manage API Keys", self)
        self.manage_keys_button.clicked.connect(manage_keys_callback)
        main_layout.addWidget(self.manage_keys_button)
        main_layout.addStretch()

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def _populate_offline_combo(self):
        import local_vlm
        current_idx = max(0, self.offline_combo.currentIndex())
        self.offline_combo.clear()
        for m in LOCAL_MODELS:
            # The size is part of the name now, so repeating it here would just make
            # the line longer to listen to.
            state = "Downloaded" if local_vlm.is_model_downloaded(m) else "Not downloaded"
            self.offline_combo.addItem(f"{m['model_name']} — {state}")
        if self.offline_combo.count() > current_idx:
            self.offline_combo.setCurrentIndex(current_idx)

    def _update_action_button(self):
        import local_vlm
        idx = self.offline_combo.currentIndex()
        if 0 <= idx < len(LOCAL_MODELS) and local_vlm.is_model_downloaded(LOCAL_MODELS[idx]):
            self.action_button.setText("Delete Selected Offline Model")
        else:
            self.action_button.setText("Download Selected Offline Model")

    def _announce(self, text: str):
        """Updates the status label and pushes a screen reader announcement without moving focus."""
        self.download_status_label.setText(text)
        announce_for_screen_reader(self.download_status_label, text)

    def on_action_button_clicked(self):
        import local_vlm
        idx = self.offline_combo.currentIndex()
        if idx < 0 or idx >= len(LOCAL_MODELS):
            return
        model = LOCAL_MODELS[idx]
        if local_vlm.is_model_downloaded(model):
            self.on_delete_clicked(model)
        else:
            self.on_download_clicked(model)

    def on_download_clicked(self, model):
        # Close the Settings dialog and let the caller run the download so that the
        # download dialog is the only open dialog.
        self.pending_download_model = model
        self.accept()

    def on_delete_clicked(self, model):
        is_current = (self.current_model_id == model['model_id'])
        dialog = DeleteConfirmDialog(model, is_current, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            import local_vlm
            ok, msg = local_vlm.delete_model(model)
            self._populate_offline_combo()
            self._update_action_button()
            if ok and is_current:
                self.model_deleted_while_active = model
            self._announce(msg)
        else:
            self._announce("Deletion cancelled. The model was kept.")

    def get_selected_language(self):
        return self.lang_combo.currentText()

    def get_tts_data(self):
        return self.voice_combo.currentText(), (self.speed_slider.value() / 100.0)