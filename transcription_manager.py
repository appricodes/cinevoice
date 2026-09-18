"""
Voice-driven query framework.
Captures live microphone audio and turns recognized speech directly into a
custom vision prompt.
"""
import threading
from PySide6.QtCore import QObject
from PySide6.QtMultimedia import QMediaPlayer
import speech_recognition as sr

from utils import get_speech_lang_code
from ui_components import CustomPromptPanel

class TranscriptionManager(QObject):
    """Coordinates microphone capture and speech-to-text for voice-driven prompts."""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.is_listening = False

    def reset_state(self):
        pass

    def start_voice_query(self):
        """Initiates local microphone recording for zero-UI multimodal query generation."""
        if self.mw.vision_manager._inflight or self.is_listening: return
        self.is_listening = True
        if self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.mw.player.pause()
        self.mw.speak("Listening... Please speak your question now.")
        threading.Thread(target=self._voice_listen_worker, daemon=True).start()
    
    def _voice_listen_worker(self):
        """Captures hardware audio streams and translates speech directly into scene analysis prompts."""
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
                self.mw.updateStatus.emit("Processing voice...")
                lang_code = get_speech_lang_code(self.mw.current_language)
                text = recognizer.recognize_google(audio, language=lang_code)
                self.mw.updateStatus.emit(f"Recognized: {text}")
                
                prompt_data = {
                    "title": "custom", "prompt": text,
                    "frames_count": CustomPromptPanel.last_frames_val,
                    "frames_interval": CustomPromptPanel.last_interval_val,
                    "max_words": CustomPromptPanel.last_words_val,
                    "is_question": True
                }
                cur_ms = self.mw.player.position()
                if cur_ms is not None and cur_ms >= 0:
                    self.mw.vision_manager.execute_prompt(cur_ms, prompt_data)
        except Exception as e:
            self.mw.updateStatus.emit(f"Voice Error: {e}")
        finally:
            self.is_listening = False