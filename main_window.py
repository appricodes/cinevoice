"""
The main window: video playback UI, all keyboard shortcuts, and TTS/screen-reader
announcements. Delegates AI description work to VisionManager and voice-query
input to TranscriptionManager.
"""
import os
import sys
import webbrowser
from PySide6 import QtCore, QtGui
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QFileDialog, QLabel, QDialog
from PySide6.QtCore import Qt, QTimer, Signal, Slot, QSettings
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtTextToSpeech import QTextToSpeech

import settings
from config import SEEK_MS, MODELS, MEDIA_EXTS, MEDIA_FILE_DIALOG_FILTER
from utils import clean_string
from api_client import MultiClient
from ui_components import ClickableSlider, SettingsDialog, LastOutputDialog, CommandMenuDialog, ModelMenuDialog, ApiKeyDialog, DownloadProgressDialog, CustomPromptPanel

from transcription_manager import TranscriptionManager
from vision_manager import VisionManager

class VideoPlayerWidget(QWidget):
    updateStatus = Signal(str)
    videoEnded = Signal()
    
    def __init__(self, media_path, api_keys, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.api_client = MultiClient(api_keys)
        self.api_client.status_cb = self.updateStatus.emit
        
        self.setWindowTitle(f"Cinevoice - {os.path.basename(self.media_path)}" if self.media_path else "Cinevoice")
        self.setGeometry(100, 100, 960, 820)
        self.q_settings = QSettings()
        
        self.custom_prompt_f7 = None
        self.custom_prompt_f8 = None
        
        self.prompts_list = settings.PROMPTS
        if not self.prompts_list:
            self.prompts_list = [{"title": "1: Default", "prompt": "Describe this scene.", "frames_count": 1, "frames_interval": 0.5}]
        self.current_prompt_data = self.prompts_list[0]
        
        # The writing style is remembered between runs, like the language and the model.
        default_mode = settings.MODES[0] if settings.MODES else "Family"
        saved_mode = self.q_settings.value("current_mode", default_mode)
        self.current_mode = saved_mode if saved_mode in settings.MODES else default_mode
        default_lang = settings.LANGUAGES[0] if settings.LANGUAGES else "English"
        self.current_language = self.q_settings.value("current_language", default_lang)
        self.current_max_words = 120

        # Ctrl+C: whether multiple captured frames are tiled into 3x3 grid images (fewer,
        # denser requests) or sent one image per frame as the app always used to. On by
        # default; remembered between runs like the model and language.
        self.grid_frames_enabled = self.q_settings.value("grid_frames_enabled", True, type=bool)
        
        self.tts = QTextToSpeech(self)
        self.available_voices = [v.name() for v in self.tts.availableVoices()]
        
        saved_voice = self.q_settings.value("tts_voice", "")
        for v in self.tts.availableVoices():
            if v.name() == saved_voice:
                self.tts.setVoice(v)
                break
                
        saved_speed = float(self.q_settings.value("tts_speed", 0.0))
        self.tts.setRate(saved_speed)
        
        saved_model_id = self.q_settings.value("current_model_id", MODELS[0]["model_id"])
        self.current_model_dict = MODELS[0]
        for m in self.get_available_models():
            if m["model_id"] == saved_model_id:
                self.current_model_dict = m
                break

        # A local model at startup gets a short default description length; it is slow.
        if self.current_model_dict.get("provider_id") == "local":
            self.current_max_words = self.local_default_words(self.current_model_dict)

        self.last_spoken_text = ""
        self.current_background_task = "Ready."
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(6, 6, 6, 6)
        
        self.video_widget = QVideoWidget(self)
        self.video_widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.video_widget.setProperty("accessibleLiveRegion", "assertive")
        self.main_layout.addWidget(self.video_widget, 1)
        
        self.progress_layout = QHBoxLayout()
        self.time_elapsed_label = QLabel("00:00")
        self.time_remaining_label = QLabel("-00:00")
        self.time_elapsed_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        self.time_remaining_label.setStyleSheet("font-family: monospace; font-size: 14px;")
        
        self.progress_slider = ClickableSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 0)
        self.progress_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        
        self.progress_layout.addWidget(self.time_elapsed_label)
        self.progress_layout.addWidget(self.progress_slider)
        self.progress_layout.addWidget(self.time_remaining_label)
        self.main_layout.addLayout(self.progress_layout, 0)

        # Embedded, hidden-until-needed instead of a separate QDialog: see CustomPromptPanel's
        # docstring for why (it's about screen readers re-announcing the main window's title).
        self.custom_prompt_panel = CustomPromptPanel(self)
        self.main_layout.addWidget(self.custom_prompt_panel, 0)

        self.setLayout(self.main_layout)
        
        self.audio_output = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.audio_output.setVolume(0.5)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        
        self.player.positionChanged.connect(self.on_position_changed)
        self.player.durationChanged.connect(self.on_duration_changed)
        self.player.mediaStatusChanged.connect(self._check_end_reached)
        self.progress_slider.sliderMoved.connect(self.player.setPosition)
        self.progress_slider.sliderClicked.connect(self.player.setPosition)
        self.updateStatus.connect(self.on_update_status)
        self.videoEnded.connect(self.handle_video_ended)

        self.transcription_manager = TranscriptionManager(self)
        self.vision_manager = VisionManager(self)
        
        if self.media_path:
            self.load_video(self.media_path)
            
    def announce_startup_model(self):
        self.speak(f"Ready. Current model is {self.current_model_dict['model_name']}.")

    def open_api_key_manager(self):
        dialog = ApiKeyDialog(self.api_client.keys, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_keys = dialog.get_keys()
            from auth import save_api_keys, load_api_keys
            ok, message = save_api_keys(new_keys)
            self.api_client.keys = load_api_keys()
            self.speak("API keys updated." if ok else message)

    def get_available_models(self) -> list:
        """Returns online models plus any offline models that are fully downloaded."""
        import local_vlm
        return MODELS + local_vlm.get_downloaded_local_models()

    @staticmethod
    def local_default_words(model_dict: dict) -> int:
        """Short default description length for a slow local model: 5 + size_gb * 5 words."""
        return int(round(5 + model_dict.get("size_gb", 0) * 5))

    def max_words_cap(self) -> int:
        """Upper limit reachable with Ctrl+Page Up. Local models cap at 10 + size_gb * 20 words."""
        m = self.current_model_dict
        if m.get("provider_id") == "local":
            return int(round(10 + m.get("size_gb", 0) * 20))
        return 2560

    def set_current_model(self, model_dict: dict, announce: bool = True):
        self.current_model_dict = model_dict
        self.q_settings.setValue("current_model_id", model_dict["model_id"])
        if model_dict.get("provider_id") == "local":
            # Slow model: start from a short default description length.
            self.current_max_words = self.local_default_words(model_dict)
        if not announce:
            return
        if model_dict.get("provider_id") == "local":
            self.speak(f"Switched model to {model_dict['model_name']}. "
                       "Warning: this offline model runs on your own computer. It is slow, and it is not suitable "
                       "for describing videos with continuous narration. Use it for single frame descriptions.")
        else:
            self.speak(f"Switched model to {model_dict['model_name']}.")

    def get_data_dir(self) -> str:
        """
        Resolves this video's private data folder next to the file, without creating it --
        callers that are about to write should create it themselves right before writing, so
        merely opening a video never leaves a folder behind for videos with nothing cached.
        Older versions used '.data_files'; if that exists but '.cinevoice' doesn't yet, it's
        renamed in place so caches already on disk aren't orphaned.
        """
        if not self.media_path: return ""
        parent = os.path.dirname(os.path.abspath(self.media_path))
        data_dir = os.path.join(parent, ".cinevoice")
        if not os.path.exists(data_dir):
            old_dir = os.path.join(parent, ".data_files")
            if os.path.exists(old_dir):
                try:
                    os.rename(old_dir, data_dir)
                except Exception:
                    pass
        return data_dir

    def get_data_file_path(self, suffix: str) -> str:
        """Builds a path inside this video's data folder, e.g. myvideo.mp4 -> .cinevoice/myvideo<suffix>, for per-video caches."""
        if not self.media_path: return ""
        base_name = os.path.splitext(os.path.basename(self.media_path))[0]
        return os.path.join(self.get_data_dir(), base_name + suffix)
    
    def load_video(self, new_path: str):
        """Stops the current video, clears its AI conversation/narration state, then loads and plays new_path."""
        self.player.stop()
        if self.vision_manager.is_auto_describing:
            self.vision_manager.auto_describe_timer.stop()
            self.vision_manager.is_auto_describing = False
            self.vision_manager._inflight = False
        
        self.transcription_manager.reset_state()
        self.vision_manager.lookahead_mode = False
        with self.vision_manager._history_lock:
            self.vision_manager.chat_history.clear()
        
        self.media_path = new_path
        self.setWindowTitle(f"Cinevoice - {os.path.basename(self.media_path)}")
        self.player.setSource(QtCore.QUrl.fromLocalFile(self.media_path))
        self.player.setPosition(0)
        
        self.vision_manager.init_lookahead_data()

        self.player.play()
        self.video_widget.setFocus()
    
    def open_and_play(self):
        last_dir = self.q_settings.value("last_directory", os.path.dirname(self.media_path) if self.media_path else ".")
        path, _ = QFileDialog.getOpenFileName(self, "Select video or image file", last_dir, MEDIA_FILE_DIALOG_FILTER)
        if path:
            self.q_settings.setValue("last_directory", os.path.dirname(path))
            self.load_video(path)
    
    def show_command_menu(self):
        was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if was_playing: self.player.pause()

        commands_dict = {
            "Open Video (Ctrl+O)": self.open_and_play,
            "Select AI Model (Ctrl+M)": lambda: self.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier)),
            "Settings (Ctrl+S)": lambda: self.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)),
            "Continuous Narrate (Ctrl+D)": self.vision_manager.toggle_lookahead_mode,
            "Batch Look-Ahead 30s Blocks (Ctrl+Shift+D)": self.vision_manager.start_batch_lookahead,
            "Full Video Cinematic Story (Shift+F12)": lambda: self.vision_manager.generate_full_video_story(force_regenerate=False),
            "Ask Voice Question (Ctrl+Shift+A)": self.transcription_manager.start_voice_query,
            "Ask Text Question (Ctrl+A)": self.vision_manager.show_custom_prompt_dialog,
            "Repeat Last Output (Ctrl+R)": lambda: self.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.ControlModifier)),
            "Toggle Frame Grid Batching (Ctrl+C)": lambda: self.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)),
            "Announce Last Call Token Usage (Ctrl+P)": lambda: self.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key.Key_P, Qt.KeyboardModifier.ControlModifier)),
            "Open Help Manual (Ctrl+H)": lambda: self.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.KeyPress, Qt.Key.Key_H, Qt.KeyboardModifier.ControlModifier))
        }

        dialog = CommandMenuDialog(commands_dict, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = dialog.get_selected()
            if selected and selected in commands_dict:
                QTimer.singleShot(50, commands_dict[selected])

        if was_playing: self.player.play()
        self.video_widget.setFocus()
    
    def navigate_folder(self, direction: int):
        if not self.media_path:
            self.speak("No video currently open.")
            return
        try:
            current_dir = os.path.dirname(os.path.abspath(self.media_path))
            current_file = os.path.basename(self.media_path)
            files = sorted([f for f in os.listdir(current_dir) if os.path.splitext(f)[1].lower() in MEDIA_EXTS])
            if current_file in files:
                new_idx = files.index(current_file) + direction
                if 0 <= new_idx < len(files):
                    self.load_video(os.path.join(current_dir, files[new_idx]))
                else:
                    self.speak("Start of folder reached." if direction == -1 else "End of folder reached.")
        except Exception:
            self.speak("Error navigating folder.")
    
    @staticmethod
    def format_time(ms: int) -> str:
        s = ms // 1000
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
    
    @Slot(int)
    def on_duration_changed(self, duration: int):
        self.progress_slider.setRange(0, duration)
        self.time_remaining_label.setText("-" + self.format_time(duration))
    
    @Slot(int)
    def on_position_changed(self, pos_ms: int):
        if not self.progress_slider.isSliderDown():
            self.progress_slider.setValue(pos_ms)
        self.time_elapsed_label.setText(self.format_time(pos_ms))
        self.time_remaining_label.setText("-" + self.format_time(max(0, self.player.duration() - pos_ms)))
        
        if self.vision_manager.lookahead_mode:
            block_idx = pos_ms // 30000
            if block_idx != self.vision_manager.current_playing_block:
                self.vision_manager.current_playing_block = block_idx
                if block_idx in self.vision_manager.lookahead_data:
                    self.speak(self.vision_manager.lookahead_data[block_idx])
                    self.vision_manager.check_and_fetch_lookahead()
                else:
                    if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                        self.player.pause()
                        self.updateStatus.emit("Buffering description for next 30 seconds...")
                        self.speak("Waiting for narrative context...")
                        self.vision_manager.waiting_for_block = block_idx
                        # Ask for it. Seeking into a stretch where the current and next
                        # blocks were both already cached leaves no fetch running, and
                        # nothing else would ever start one -- the video would sit paused.
                        self.vision_manager.check_and_fetch_lookahead()
    
    def _check_end_reached(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.videoEnded.emit()
    
    @Slot(str)
    def on_update_status(self, message: str):
        self.current_background_task = clean_string(message)
    
    @Slot()
    def handle_video_ended(self):
        self.player.stop()
        self.player.setPosition(0)
        self.video_widget.setFocus()
        
    def speak(self, text: str):
        """
        Announces text to the user: via the active screen reader (NVDA/JAWS/Narrator) if one
        is running, otherwise via the app's own built-in TTS voice.
        """
        text = clean_string(text)
        self.last_spoken_text = text
        
        if not self.video_widget.hasFocus():
            self.video_widget.setFocus()
            
        if not QtGui.QAccessible.isActive():
            self.tts.say(text)
            return
        else:
            if self.tts.state() == QTextToSpeech.State.Speaking:
                self.tts.stop()
        
        # Prefer the modern Announcement event (reads text without needing a name change);
        # QAccessibleAnnouncementEvent doesn't exist on older Qt/PySide6 versions, so fall
        # back to setting the accessible name and firing an Alert event, which NVDA also picks up.
        announcement_event = getattr(QtGui, "QAccessibleAnnouncementEvent", None)
        if announcement_event:
            try:
                ev = announcement_event(self.video_widget, text)
                QtGui.QAccessible.updateAccessibility(ev)
                return
            except:
                pass

        self.video_widget.setAccessibleName(text)
        try:
            alert_enum = getattr(QtGui.QAccessible.Event, "Alert", None)
            if alert_enum is None:
                alert_enum = getattr(QtGui.QAccessible, "Alert", None)
            if alert_enum is not None:
                alert_ev = QtGui.QAccessibleEvent(self.video_widget, alert_enum)
                QtGui.QAccessible.updateAccessibility(alert_ev)
        except Exception:
            pass

    def keyPressEvent(self, event):
        if self.tts.state() == QTextToSpeech.State.Speaking:
            self.tts.stop()

        key = event.key()
        modifiers = event.modifiers()
        has_ctrl = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
        has_shift = bool(modifiers & Qt.KeyboardModifier.ShiftModifier)
        has_alt = bool(modifiers & Qt.KeyboardModifier.AltModifier)
        
        handled = True 
        
        if has_ctrl and has_shift and not has_alt:
            if key == Qt.Key.Key_A:
                self.transcription_manager.start_voice_query()
            elif key == Qt.Key.Key_F12:
                self.vision_manager.generate_full_video_story(force_regenerate=True)
            elif key == Qt.Key.Key_D:
                self.vision_manager.start_batch_lookahead()
            else:
                handled = False
        
        elif has_ctrl and not has_shift and not has_alt:
            if key == Qt.Key.Key_M:
                available_models = self.get_available_models()
                dialog = ModelMenuDialog(available_models, self.current_model_dict["model_id"], self.api_client.keys, self)
                was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                if was_playing: self.player.pause()

                if dialog.exec() == QDialog.DialogCode.Accepted:
                    idx = dialog.get_selected_index()
                    if 0 <= idx < len(available_models):
                        selected_model = available_models[idx]
                        if selected_model.get("provider_id") == "local":
                            self.set_current_model(selected_model)
                        elif not self.api_client.keys.get(selected_model["provider_id"]):
                            self.speak("API key required. Opening key manager.")
                            self.open_api_key_manager()
                            if self.api_client.keys.get(selected_model["provider_id"]):
                                self.set_current_model(selected_model)
                        else:
                            self.set_current_model(selected_model)
                            
                if was_playing: self.player.play()
                self.video_widget.setFocus()
            elif key == Qt.Key.Key_PageUp:
                self.current_max_words = min(self.max_words_cap(), max(1, self.current_max_words) * 2)
                self.speak(f"Description length set to {self.current_max_words} words.")
            elif key == Qt.Key.Key_PageDown:
                self.current_max_words = max(10, self.current_max_words // 2)
                self.speak(f"Description length set to {self.current_max_words} words.")
            elif key == Qt.Key.Key_Up:
                self.vision_manager.trigger_current_prompt()
            elif key == Qt.Key.Key_Down:
                self.show_command_menu()
            elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
                idx = key - Qt.Key.Key_F1
                if 0 <= idx < len(settings.MODES):
                    self.current_mode = settings.MODES[idx]
                    self.q_settings.setValue("current_mode", self.current_mode)
                    self.speak(self.current_mode)
            elif key == Qt.Key.Key_A:
                self.vision_manager.show_custom_prompt_dialog()
            elif key == Qt.Key.Key_S:
                was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                if was_playing: self.player.pause()
                
                cur_voice = self.tts.voice().name() if self.tts.voice() else ""
                cur_speed = self.tts.rate()
                dialog = SettingsDialog(settings.LANGUAGES, self.current_language, self.available_voices, cur_voice, cur_speed, self.open_api_key_manager, self.current_model_dict["model_id"], self)
                
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    self.current_language = dialog.get_selected_language()
                    self.q_settings.setValue("current_language", self.current_language)

                    new_voice_name, new_speed = dialog.get_tts_data()
                    self.q_settings.setValue("tts_voice", new_voice_name)
                    self.q_settings.setValue("tts_speed", new_speed)

                    self.tts.setRate(new_speed)
                    for v in self.tts.availableVoices():
                        if v.name() == new_voice_name:
                            self.tts.setVoice(v)
                            break

                # If the user chose to download an offline model, the Settings
                # dialog has already closed. Run the download in its own dialog so
                # it is the only one open. On success it becomes the current model
                # (the success dialog already announced it, so switch quietly).
                if dialog.pending_download_model is not None:
                    dl = DownloadProgressDialog(dialog.pending_download_model, self)
                    if dl.exec() == QDialog.DialogCode.Accepted and dl.success:
                        self.set_current_model(dialog.pending_download_model, announce=False)

                # The Settings dialog deletes offline models synchronously on disk,
                # independent of its own OK/Cancel result, so check regardless.
                if dialog.model_deleted_while_active is not None:
                    deleted = dialog.model_deleted_while_active
                    self.set_current_model(MODELS[0], announce=False)
                    self.speak(f"{deleted['model_name']} was deleted from your computer. "
                              f"Switched back to {MODELS[0]['model_name']}.")

                if was_playing: self.player.play()
            elif key == Qt.Key.Key_D:
                self.vision_manager.toggle_lookahead_mode()
            elif key == Qt.Key.Key_F:
                if not self.vision_manager.is_auto_describing:
                    self.speak("Starting auto-describe.")
                    self.vision_manager.is_auto_describing = True
                    self.vision_manager.auto_describe_timer.start()
                    self.vision_manager.on_auto_describe_tick()
                else:
                    self.speak("Stopping auto-describe.")
                    self.vision_manager.is_auto_describing = False
                    self.vision_manager.auto_describe_timer.stop()
            elif key == Qt.Key.Key_O:
                self.open_and_play()
            elif key == Qt.Key.Key_H:
                # PyInstaller's onedir build extracts bundled data files (including
                # help.html) into sys._MEIPASS, which is the "_internal" folder next
                # to the exe, not the exe's own folder.
                if hasattr(sys, '_MEIPASS'):
                    base_dir = sys._MEIPASS
                else:
                    base_dir = os.path.dirname(os.path.abspath(__file__))

                help_path = os.path.join(base_dir, "help.html")
                
                if os.path.exists(help_path):
                    webbrowser.open(f"file:///{help_path.replace(os.sep, '/')}")
                    self.speak("Opened Help manual in browser.")
                else:
                    self.speak("Could not find help documentation file.")
            elif key == Qt.Key.Key_R:
                was_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
                if was_playing: self.player.pause()
                LastOutputDialog(self.last_spoken_text, self).exec()
                if was_playing: self.player.play()
            elif key == Qt.Key.Key_I:
                self.speak(f"Background Progress: {self.current_background_task}")
            elif key == Qt.Key.Key_C:
                self.grid_frames_enabled = not self.grid_frames_enabled
                self.q_settings.setValue("grid_frames_enabled", self.grid_frames_enabled)
                if self.grid_frames_enabled:
                    self.speak("Frame grid batching enabled. Multiple frames are tiled into 3x3 grid images.")
                else:
                    self.speak("Frame grid batching disabled. Frames are sent one image at a time.")
            elif key == Qt.Key.Key_P:
                info = self.api_client.last_call_info
                usage = info.get("usage") if info else None
                if not usage:
                    self.speak("No token usage information is available for the last call.")
                else:
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
                    msg = f"Last call used {prompt_tokens} input tokens and {completion_tokens} output tokens."
                    if info.get("provider_id") == "grok":
                        ticks = usage.get("cost_in_usd_ticks")
                        if ticks is not None:
                            msg += f" Cost: {ticks / 100_000_000:.4f} cents."
                    self.speak(msg)
            elif key == Qt.Key.Key_Left:
                self.player.setPosition(max(0, self.player.position() - 6 * SEEK_MS))
            elif key == Qt.Key.Key_Right:
                dur = self.player.duration() or 0
                cur = self.player.position()
                self.player.setPosition(int(min(dur, cur + 6 * SEEK_MS) if dur > 0 else cur + 6 * SEEK_MS))
            else:
                handled = False
        
        elif has_shift and not has_ctrl and not has_alt:
            if key == Qt.Key.Key_F7:
                self.vision_manager.assign_custom_prompt(7)
            elif key == Qt.Key.Key_F8:
                self.vision_manager.assign_custom_prompt(8)
            elif key == Qt.Key.Key_F12:
                self.vision_manager.generate_full_video_story(force_regenerate=False)
            else:
                handled = False
        
        elif not has_ctrl and not has_shift and not has_alt:
            if key == Qt.Key.Key_Home:
                self.player.setPosition(0)
                self.speak("Moved to beginning.")
            elif key == Qt.Key.Key_Left:
                self.player.setPosition(max(0, self.player.position() - SEEK_MS))
            elif key == Qt.Key.Key_Right:
                dur = self.player.duration() or 0
                cur = self.player.position()
                self.player.setPosition(int(min(dur, cur + SEEK_MS) if dur > 0 else cur + SEEK_MS))
            elif key == Qt.Key.Key_Up:
                new_vol = min(2.0, self.audio_output.volume() + 0.05)
                self.audio_output.setVolume(new_vol)
                self.speak(f"Volume {int(new_vol * 100)} percent")
            elif key == Qt.Key.Key_Down:
                new_vol = max(0.0, self.audio_output.volume() - 0.05)
                self.audio_output.setVolume(new_vol)
                self.speak(f"Volume {int(new_vol * 100)} percent")
            elif Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
                pct_map = {Qt.Key.Key_1: 0.0, Qt.Key.Key_2: 0.1, Qt.Key.Key_3: 0.2, Qt.Key.Key_4: 0.3, Qt.Key.Key_5: 0.4,
                           Qt.Key.Key_6: 0.5, Qt.Key.Key_7: 0.6, Qt.Key.Key_8: 0.7, Qt.Key.Key_9: 0.8, Qt.Key.Key_0: 0.9}
                if key in pct_map and self.player.duration() > 0:
                    self.player.setPosition(int(self.player.duration() * pct_map[key]))
            elif key in (Qt.Key.Key_MediaTogglePlayPause, Qt.Key.Key_MediaPlay, Qt.Key.Key_MediaPause, Qt.Key.Key_Space):
                self.player.pause() if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState else self.player.play()
            elif key == Qt.Key.Key_PageUp:
                self.navigate_folder(-1)
            elif key == Qt.Key.Key_PageDown:
                self.navigate_folder(1)
            elif Qt.Key.Key_F1 <= key <= Qt.Key.Key_F12:
                if key == Qt.Key.Key_F7:
                    if self.custom_prompt_f7:
                        self.current_prompt_data = self.custom_prompt_f7
                        self.vision_manager.trigger_current_prompt()
                    else:
                        self.speak("Custom 1. Press Shift plus F7 to assign.")
                elif key == Qt.Key.Key_F8:
                    if self.custom_prompt_f8:
                        self.current_prompt_data = self.custom_prompt_f8
                        self.vision_manager.trigger_current_prompt()
                    else:
                        self.speak("Custom 2. Press Shift plus F8 to assign.")
                else:
                    idx = (key - Qt.Key.Key_F1) if key <= Qt.Key.Key_F6 else (len(self.prompts_list) - 4 + (key - Qt.Key.Key_F9))
                    if 0 <= idx < len(self.prompts_list):
                        self.current_prompt_data = self.prompts_list[idx]
                        self.vision_manager.trigger_current_prompt()
            else:
                handled = False
        else:
            handled = False

        # Shortcuts the app acts on are consumed here; anything else goes to the base class
        # so Qt's own handling (tab order, accelerators) still works.
        if handled:
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def closeEvent(self, event):
        self.vision_manager.auto_describe_timer.stop()
        try:
            self.player.stop()
        except:
            pass
        event.accept()