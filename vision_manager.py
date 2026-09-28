"""
Captures video frames and sends them to the AI model for description, plus the
supporting features built on top of that: continuous 30-second narration, batch
pre-generation, and full-video story summaries.
"""
import os
import json
import threading
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QDialog
from PySide6.QtMultimedia import QMediaPlayer

from config import MAX_IMAGE_DIM, XAI_MAX_IMAGE_DIM, FRAME_GRID_SIZE, MP4_METADATA_EXTS
from utils import clean_string, resize_to_max_dim, encode_frame, build_frame_grids
from ui_components import ExistingDescriptionDialog
import mp4_metadata

class VisionManager(QObject):
    """Owns all frame-capture and AI-description logic; one instance per open video."""
    grokResponseReady = Signal(object, bool)  # named after the original provider; now fires for any model
    lookaheadBlockReady = Signal(int, str)
    batchCompleteSignal = Signal()
    fullStoryReady = Signal(str)
    streamChunkReady = Signal(str)
    lookaheadFailed = Signal(int, str)

    # How many messages of question/answer context travel with each request. Without a cap
    # the history grows for as long as the app is open -- auto-describe alone adds a turn
    # every five seconds -- and every request re-sends all of it, so cost and latency climb
    # with no upper bound.
    MAX_HISTORY_MESSAGES = 12

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.chat_history = []
        self._history_lock = threading.Lock()
        self._embedded_meta_lock = threading.Lock()
        self._inflight = False
        self._batch_inflight = False

        self.lookahead_mode = False
        self.lookahead_data = {}
        self.lookahead_json_path = ""
        self.current_playing_block = -1
        self.fetching_block = -1
        self.waiting_for_block = -1
        
        self.is_auto_describing = False
        self.auto_describe_timer = QTimer(self)
        self.auto_describe_timer.setInterval(5000)
        self.auto_describe_timer.timeout.connect(self.on_auto_describe_tick)

        # Tracks what a currently-open custom_prompt_panel is for: None means a plain
        # question (Ctrl+A), an int means "assign to this F-key slot" (assign_custom_prompt).
        # Only one can be open at a time, so one pair of fields is enough to remember it
        # between open_for_input() and the panel's accepted/cancelled signal firing back.
        self._pending_prompt_slot = None
        self._prompt_was_playing = False
        self.mw.custom_prompt_panel.accepted.connect(self._on_custom_prompt_accepted)
        self.mw.custom_prompt_panel.cancelled.connect(self._on_custom_prompt_cancelled)
        self._story_was_playing = False
        self.mw.story_panel.closed.connect(self._on_story_panel_closed)

        self.grokResponseReady.connect(self.handle_grok_response)
        self.lookaheadBlockReady.connect(self.handle_lookahead_ready)
        self.batchCompleteSignal.connect(self.on_batch_complete)
        self.fullStoryReady.connect(self.handle_full_story_ready)
        self.streamChunkReady.connect(self.handle_stream_chunk)
        self.lookaheadFailed.connect(self.handle_lookahead_failed)

    def _append_history(self, role: str, content: str):
        """Records one conversation turn, keeping only the most recent MAX_HISTORY_MESSAGES."""
        with self._history_lock:
            self.chat_history.append({"role": role, "content": content})
            if len(self.chat_history) > self.MAX_HISTORY_MESSAGES:
                del self.chat_history[:-self.MAX_HISTORY_MESSAGES]

    def start_prompt(self, timestamp_ms: int, prompt_data: dict, announce_busy: bool = True) -> bool:
        """
        Runs one prompt on a worker thread, unless one is already running.

        _inflight is set here, before the thread starts, rather than inside the worker: set
        from the worker it would already be too late, and a second key press could slip past
        the check and have two answers spoken over each other.
        """
        if self._inflight:
            if announce_busy:
                self.mw.speak("Still working on the previous request.")
            return False
        self._inflight = True

        def run():
            try:
                self.execute_prompt(timestamp_ms, prompt_data)
            finally:
                self._inflight = False

        threading.Thread(target=run, daemon=True).start()
        return True

    def _is_local_model(self) -> bool:
        return self.mw.current_model_dict.get("provider_id") == "local"

    def _is_xai_model(self) -> bool:
        return self.mw.current_model_dict.get("provider_id") == "grok"

    def _prepare_frames_for_api(self, raw_frames: list) -> tuple:
        """
        Turns freshly captured OpenCV frames into the final base64 JPEG images to send.

        When more than one frame is captured and frame-grid batching (Ctrl+C) is on, frames
        are tiled into as few images as possible -- up to FRAME_GRID_SIZE^2 per image, e.g. 90
        frames as 10 images of a 3x3 grid each -- instead of one image per frame. Off, or with
        a single frame, each frame is sent as its own image exactly as before.

        A Grok model additionally gets every outgoing image (grid or not) downscaled to
        XAI_MAX_IMAGE_DIM, since that's the tile size its vision encoder uses internally.

        Returns (frames_b64, grid_note): grid_note explains the tiling to the model and
        should be appended to the system prompt, or is "" when no grid was used.
        """
        thumbnails = [resize_to_max_dim(f, MAX_IMAGE_DIM) for f in raw_frames]

        grid_note = ""
        if len(thumbnails) > 1 and self.mw.grid_frames_enabled:
            cells_per_image = FRAME_GRID_SIZE * FRAME_GRID_SIZE
            images = build_frame_grids(thumbnails, cells_per_image)
            grid_note = (
                f"Note on image format: the {len(thumbnails)} video frames (oldest first) were combined "
                f"into {len(images)} image(s) instead of sending one image per frame. Each image tiles up "
                f"to {cells_per_image} frames in a {FRAME_GRID_SIZE}x{FRAME_GRID_SIZE} grid, read left to "
                f"right, then top to bottom, oldest frame first; any unused cells at the end of the last "
                f"image are blank. Treat every tile as one consecutive moment in time, not a separate picture."
            )
        else:
            images = thumbnails

        if self._is_xai_model():
            images = [resize_to_max_dim(img, XAI_MAX_IMAGE_DIM) for img in images]

        frames_b64 = [encode_frame(img, quality=90) for img in images]
        return frames_b64, grid_note

    def _uses_embedded_metadata(self) -> bool:
        """Whether the current video's container can hold our descriptions directly."""
        if not self.mw.media_path: return False
        return os.path.splitext(self.mw.media_path)[1].lower() in MP4_METADATA_EXTS

    def _update_embedded_metadata(self, **updates) -> bool:
        """Read-modify-write the embedded metadata box so a block-description save doesn't
        clobber the full story (and vice versa) -- both live in the same box."""
        with self._embedded_meta_lock:
            data = mp4_metadata.read_metadata(self.mw.media_path)
            data.update(updates)
            return mp4_metadata.write_metadata(self.mw.media_path, data)

    def _migrate_legacy_sidecars_to_embedded(self):
        """
        One-time migration: this video may still have old sidecar description files
        sitting in its .cinevoice folder (from before descriptions moved into the MP4
        itself, or before that folder was still named .data_files -- get_data_dir already
        renames that in place). If so, fold them into the video's embedded metadata and
        delete the sidecar files, but only after the embed actually succeeds.
        """
        if not self._uses_embedded_metadata(): return
        json_path = self.mw.get_data_file_path(".json")
        story_path = self.mw.get_data_file_path(".story.txt")
        if not os.path.exists(json_path) and not os.path.exists(story_path):
            return

        updates = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    updates["blocks"] = json.load(f)
            except Exception:
                pass
        if os.path.exists(story_path):
            try:
                with open(story_path, 'r', encoding='utf-8') as f:
                    updates["story"] = f.read()
            except Exception:
                pass
        if not updates or not self._update_embedded_metadata(**updates):
            return
        for path in (json_path, story_path):
            try:
                if os.path.exists(path): os.remove(path)
            except Exception:
                pass

    def assign_custom_prompt(self, slot):
        if self.mw.custom_prompt_panel.isVisible(): return
        self._pending_prompt_slot = slot
        self._prompt_was_playing = self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if self._prompt_was_playing: self.mw.player.pause()
        self.mw.speak(f"Assign custom prompt to F{slot}.")
        self.mw.custom_prompt_panel.open_for_input()

    def show_custom_prompt_dialog(self):
        if self.mw.custom_prompt_panel.isVisible(): return
        self._pending_prompt_slot = None
        self._prompt_was_playing = self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if self._prompt_was_playing: self.mw.player.pause()
        self.mw.speak("Ask a question.")
        self.mw.custom_prompt_panel.open_for_input()

    def _on_custom_prompt_accepted(self, prompt_text, frames_count, frames_interval, max_words):
        if self._prompt_was_playing: self.mw.player.play()
        self.mw.video_widget.setFocus()

        slot = self._pending_prompt_slot
        if slot is not None:
            prompt_data = {"title": f"Custom {slot - 6}", "prompt": prompt_text, "frames_count": frames_count,
                           "frames_interval": frames_interval, "max_words": max_words,
                           "is_question": True}
            if slot == 7:
                self.mw.custom_prompt_f7 = prompt_data
            elif slot == 8:
                self.mw.custom_prompt_f8 = prompt_data
            self.mw.current_prompt_data = prompt_data
            self.mw.speak(f"Assigned to F{slot}. Processing...")
        else:
            self.mw.current_prompt_data = {"title": "custom", "prompt": prompt_text, "frames_count": frames_count,
                                           "frames_interval": frames_interval, "max_words": max_words,
                                           "is_question": True}

        cur_ms = self.mw.player.position()
        if cur_ms is not None and cur_ms >= 0:
            self.start_prompt(cur_ms, self.mw.current_prompt_data)

    def _on_custom_prompt_cancelled(self):
        if self._prompt_was_playing: self.mw.player.play()
        self.mw.video_widget.setFocus()
    
    def trigger_current_prompt(self):
        self.mw.speak(self.mw.current_prompt_data.get('title', 'Processing'))
        self.is_auto_describing = False
        with self._history_lock: self.chat_history.clear()
        cur_ms = self.mw.player.position()
        if cur_ms is not None and cur_ms >= 0:
            self.start_prompt(cur_ms, self.mw.current_prompt_data)
    
    def _build_system_prompt(self, is_question: bool, frame_count: int, max_words: int) -> str:
        """
        Builds the system prompt for one request.

        A free-form request -- a typed question (Ctrl+A), a spoken one (Ctrl+Shift+A), or a
        custom prompt on F7/F8 -- gets a deliberately open prompt. These used to be given the
        same "cinematic audio describer" prompt as the F1-F6 description keys, whose persona
        and describe-only rules pushed the model into narrating the scene rather than
        answering what was asked, so neither survives here. The writing style does: it comes
        last and shapes only the wording, after the question itself has set the answer.
        """
        # Several frames are consecutive moments of one shot, not unrelated pictures.
        # Unsaid, the model tends to walk through them one at a time, which is why a
        # question answered badly got worse the more frames were attached to it.
        frames_note = ""
        if frame_count > 1:
            frames_note = (f"The {frame_count} images are consecutive moments from the video, "
                           f"oldest first; read them as one continuous scene. ")

        # Local models are small and quantized: a short, single-clause instruction is
        # followed far more reliably than the longer, multi-rule prompt used for the big
        # online models (language is handled separately, right before generation, by
        # local_vlm._enforce_language, so it's omitted here).
        if self._is_local_model():
            if is_question:
                return (f"{frames_note}Answer the user's question. "
                        f"Tone: {self.mw.current_mode}. Under {max_words} words.")
            return (f"Describe the scene. Style: {self.mw.current_mode}. "
                    f"Under {max_words} words. Do not say 'image' or 'frame'.")

        if is_question:
            return (f"You are watching this video with the user and answering their questions about it. "
                    f"Respond in {self.mw.current_language}. "
                    f"{frames_note}"
                    f"Answer what was actually asked, and let the question decide the shape of the answer -- "
                    f"describe the scene only when the question asks for a description. "
                    f"If the video does not show enough to answer, say so. "
                    f"Word the answer in a '{self.mw.current_mode}' tone. "
                    f"Keep it under {max_words} words.")

        return (f"You are a professional cinematic audio describer. Respond in {self.mw.current_language} using a '{self.mw.current_mode}' style. "
                f"Limit to {max_words} words. "
                f"CRITICAL INSTRUCTIONS: "
                f"1. Never use words like 'image', 'frame', 'picture', or 'shows'. Treat the visual input as a living, unfolding world. "
                f"2. Write in the immediate present tense and active voice. "
                f"3. Strictly describe ONLY what is actually, physically present in the scene. Do not hallucinate or assume unseen events.")

    def execute_prompt(self, timestamp_ms: int, prompt_data: dict):
        """Grabs the frame(s) prompt_data asks for around timestamp_ms and sends them to the AI model."""
        import cv2
        try:
            self.mw.updateStatus.emit("Capturing frame(s)...")
            prompt_text = prompt_data.get('prompt', "Describe the scene.")
            frames_count = int(prompt_data.get('frames_count', 1))
            frames_interval_ms = int(float(prompt_data.get('frames_interval', 0.5)) * 1000)
            max_words = prompt_data.get('max_words', self.mw.current_max_words)
            
            capture_times_ms = []
            if frames_count <= 1:
                capture_times_ms.append(timestamp_ms)
            else:
                for i in range(frames_count):
                    ms_time = timestamp_ms - (i * frames_interval_ms)
                    if ms_time >= 0:
                        capture_times_ms.append(ms_time)
                    else:
                        break
                capture_times_ms.sort()
            
            cap = cv2.VideoCapture(self.mw.media_path)
            raw_frames = []
            try:
                for ms_time in capture_times_ms:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(ms_time))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        raw_frames.append(frame)
            finally:
                cap.release()

            captured_frames_b64, grid_note = self._prepare_frames_for_api(raw_frames)

            final_user_prompt = prompt_text

            system_prompt_string = self._build_system_prompt(
                is_question=bool(prompt_data.get('is_question')),
                frame_count=len(raw_frames),
                max_words=max_words,
            )
            if grid_note:
                system_prompt_string += " " + grid_note

            with self._history_lock:
                # Snapshot before appending: the current question is sent on its own below,
                # as the real user message carrying the frames, so leaving it in the history
                # as well would deliver it to the model twice.
                history_snapshot = list(self.chat_history)
            self._append_history("user", final_user_prompt)

            is_local = self._is_local_model()
            # Local models are slow enough that waiting for the full response before saying
            # anything is a poor experience: stream it out clause by clause as it's generated.
            stream_cb = self.streamChunkReady.emit if is_local else None

            self.mw.updateStatus.emit(f"Sending frames to API...")
            response = self.mw.api_client.send(
                model_dict=self.mw.current_model_dict,
                frames_b64=captured_frames_b64,
                system_prompt=system_prompt_string,
                user_text_blocks=[final_user_prompt],
                expect_json=False,
                history=history_snapshot,
                language=self.mw.current_language,
                stream_cb=stream_cb
            )
            self.grokResponseReady.emit(response, is_local)
        except Exception as e:
            self.grokResponseReady.emit({"error": f"Execution failed: {e}"}, False)
    
    @Slot()
    def on_auto_describe_tick(self):
        """Runs every 5 seconds while auto-describe is on, asking only what changed since last time."""
        if not self.is_auto_describing or self.mw.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState or self._inflight: return
        cur_ms = self.mw.player.position()
        if cur_ms is None or cur_ms < 5000: return
        auto_prompt_data = {"title": "Auto-Describe-Diff", "prompt": "Describe only new changes in the scene.",
                            "frames_count": 2, "frames_interval": 5.0}
        # Silent when busy: this fires every five seconds and would otherwise nag.
        self.start_prompt(cur_ms, auto_prompt_data, announce_busy=False)
    
    @Slot(str)
    def handle_stream_chunk(self, chunk: str):
        """Announces one streamed clause/sentence from a local model as soon as it's ready."""
        self.mw.speak(chunk)

    @Slot(object, bool)
    def handle_grok_response(self, response, streamed_locally: bool = False):
        """Speaks the AI's reply (or the error) and records it in chat_history for follow-up questions."""
        desc = "Error: Unknown response failure."
        is_error = True
        already_spoken = False
        if response and isinstance(response, dict):
            if "choices" in response:
                desc = response["choices"][0]["message"]["content"].strip()
                is_error = False
                # A local model's response was already spoken piece by piece as it streamed in.
                already_spoken = streamed_locally
            elif "error" in response:
                desc = response["error"]
        desc = clean_string(desc)

        if not already_spoken:
            self.mw.speak(desc)
        else:
            # Each streamed chunk overwrote last_spoken_text on its way out, leaving only the
            # final clause there; put the whole reply back so Ctrl+R repeats all of it.
            self.mw.last_spoken_text = desc

        if is_error:
            # Keep the failure out of the conversation: as an assistant turn it would be fed
            # back as context on the next question, and the unanswered question ahead of it
            # would read as one the model had ignored.
            with self._history_lock:
                if self.chat_history and self.chat_history[-1].get("role") == "user":
                    self.chat_history.pop()
            return
        self._append_history("assistant", desc)
    
    def init_lookahead_data(self):
        """Loads any 30-second block descriptions already saved for this video: embedded in
        the file itself for MP4/MOV, or a legacy sidecar .json for other containers."""
        if not self.mw.media_path: return
        self.lookahead_data.clear()
        if self._uses_embedded_metadata():
            self._migrate_legacy_sidecars_to_embedded()
            self.lookahead_json_path = ""
            for item in mp4_metadata.read_metadata(self.mw.media_path).get("blocks", []):
                self.lookahead_data[item.get("time", 0) // 30] = item.get("description", "")
            return
        self.lookahead_json_path = self.mw.get_data_file_path(".json")
        if os.path.exists(self.lookahead_json_path):
            try:
                with open(self.lookahead_json_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        self.lookahead_data[item.get("time", 0) // 30] = item.get("description", "")
            except Exception:
                pass

    def save_lookahead_data(self):
        """Persists the 30-second block descriptions so they survive across sessions:
        embedded in the video file itself for MP4/MOV, or a legacy sidecar .json otherwise."""
        data = [{"time": idx * 30, "description": desc} for idx, desc in sorted(self.lookahead_data.items())]
        if self._uses_embedded_metadata():
            self._update_embedded_metadata(blocks=data)
            return
        if not self.lookahead_json_path: return
        try:
            os.makedirs(os.path.dirname(self.lookahead_json_path), exist_ok=True)
            with open(self.lookahead_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def toggle_lookahead_mode(self):
        if self.lookahead_mode:
            self.lookahead_mode = False
            self.mw.speak("Continuous narration disabled.")
            self.waiting_for_block = -1
            self.fetching_block = -1
            if self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                self.mw.player.play()
        else:
            self.enable_lookahead_mode()

    def enable_lookahead_mode(self):
        """Turns on continuous 30-second narration; safe to call even if already enabled."""
        self.lookahead_mode = True
        self.mw.speak("Continuous 30-second narration enabled.")
        self.is_auto_describing = False
        self.auto_describe_timer.stop()
        self.init_lookahead_data()
        self.current_playing_block = -1
        self.check_and_fetch_lookahead()
    
    def check_and_fetch_lookahead(self):
        """Fetches the current or next 30-second block's description if it isn't cached yet, staying one block ahead of playback."""
        if not self.lookahead_mode: return
        current_block_idx = self.mw.player.position() // 30000
        dur = self.mw.player.duration()
        max_blocks = (dur // 30000) + 1 if dur > 0 else 999
        target_idx = current_block_idx if current_block_idx not in self.lookahead_data else current_block_idx + 1 if (current_block_idx + 1) not in self.lookahead_data else -1
        if target_idx == -1 or target_idx == self.fetching_block or target_idx >= max_blocks: return
        
        self.fetching_block = target_idx
        threading.Thread(target=self._fetch_lookahead_worker, args=(target_idx,), daemon=True).start()
    
    def _recent_block_descriptions(self, block_idx: int, count: int = 3) -> list:
        """
        The descriptions of the blocks just before block_idx, oldest first.

        More than one is needed. With only the immediately preceding block as context the
        model has no idea what was established two blocks ago, so it re-introduces the same
        room, the same people and the same furniture every minute or so.
        """
        recent = []
        for i in range(max(0, block_idx - count), block_idx):
            desc = self.lookahead_data.get(i, "")
            if desc and not desc.startswith("Error"):
                recent.append(desc)
        return recent

    def describe_block(self, media_path: str, block_idx: int) -> str:
        """
        Describes the 30-second block at block_idx using 3 sampled frames, as live audio
        description for a blind listener: only what is new in those 30 seconds, never a
        re-description of what the preceding blocks already said. The descriptions of the
        blocks just before this one are sent along so the model knows what is already known.
        """
        import cv2
        start_ms = block_idx * 30000
        self.mw.updateStatus.emit(f"Extracting frames for 30s block {block_idx}...")
        raw_frames = []
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            return f"Error: Cannot open video file {os.path.basename(media_path)}"
        try:
            for i in range(3):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms + (i * 10000)))
                ok, frame = cap.read()
                if ok and frame is not None:
                    raw_frames.append(frame)
                else:
                    break
        finally:
            cap.release()

        if not raw_frames: return "Error: No frames extracted."
        frames_b64, grid_note = self._prepare_frames_for_api(raw_frames)

        prev_descs = self._recent_block_descriptions(block_idx)

        # This is audio description, not storytelling. The old prompt asked for "just enough
        # scene detail -- setting, mood, who's involved -- to help the listener picture it"
        # on every single block, which is the instruction that made the man sit down on the
        # same wooden chair again every 30 seconds. What the listener needs is only what
        # changed since the last block, plus permission to say little when little happens.
        if self._is_local_model():
            system_prompt = (
                f"Audio description for a blind listener. Say only what is new or what changes in these frames. "
                f"Do not repeat anything already described. Tone: {self.mw.current_mode}. "
                f"Under {self.mw.current_max_words} words. Do not say 'image' or 'frame'."
            )
            user_text = "What happens in these 30 seconds?"
            # Small local models lose the thread in long context: give them the last block only.
            if prev_descs:
                user_text += f" Already described, do not repeat: {prev_descs[-1]}"
        else:
            system_prompt = (
                f"You are giving live audio description of a video to a blind listener, in {self.mw.current_language}. "
                f"These {len(raw_frames)} frames are the next 30 seconds, in chronological order. "
                f"Limit to {self.mw.current_max_words} words. "
                f"CRITICAL INSTRUCTIONS: "
                f"1. Report only what is NEW in these 30 seconds: what happens, what changes, anyone or anywhere not introduced yet, and any text that appears on screen. "
                f"2. The listener has already heard every earlier description. Never restate it. A person, place or object they already know is referred to in a word or two, never described again. "
                f"3. Describe a setting in full only the first time it appears, or when it genuinely changes. "
                f"4. Write in the immediate present tense and active voice, and never use words like 'image', 'frame' or 'shows'. "
                f"5. Skip micro-movements -- a glance, a small gesture, a shift in posture -- unless they matter to what is happening. "
                f"6. If little changes in these 30 seconds, say so in a few words. A short description is correct; never pad it out by re-describing the scene. "
                f"7. Word all of this in a '{self.mw.current_mode}' tone. The tone changes how it is worded, never what gets reported -- it can never justify re-describing the scene or padding a short description."
            )
            user_text = "Describe what happens in these 30 seconds."
            if prev_descs:
                already = "\n\n".join(prev_descs)
                user_text += (f"\n\nALREADY DESCRIBED TO THE LISTENER, oldest first -- none of this may be repeated:\n"
                              f"\"\"\"\n{already}\n\"\"\"")
        if grid_note:
            system_prompt += " " + grid_note

        try:
            self.mw.updateStatus.emit(f"Sending block {block_idx} to API...")
            response = self.mw.api_client.send(
                model_dict=self.mw.current_model_dict,
                frames_b64=frames_b64,
                system_prompt=system_prompt,
                user_text_blocks=[user_text],
                expect_json=False,
                language=self.mw.current_language
            )
            if response and isinstance(response, dict):
                if "choices" in response:
                    return clean_string(response["choices"][0]["message"]["content"].strip())
                elif "error" in response:
                    return f"Error: {response['error']}"
            return "Error: Failed to describe block."
        except Exception as e:
            return f"Error during block description: {e}"

    def _fetch_lookahead_worker(self, block_idx: int):
        desc = self.describe_block(self.mw.media_path, block_idx)
        if desc.startswith("Error"):
            self.fetching_block = -1
            self.lookaheadFailed.emit(block_idx, desc)
        else:
            self.lookaheadBlockReady.emit(block_idx, desc)

    @Slot(int, str)
    def handle_lookahead_failed(self, block_idx: int, message: str):
        """
        A block's description could not be produced. If playback is paused waiting for
        exactly that block, say what went wrong and let the video run again -- otherwise it
        would sit frozen with nothing left on its way to un-freeze it.
        """
        if self.waiting_for_block != block_idx:
            return
        self.waiting_for_block = -1
        self.mw.speak(clean_string(message))
        self.mw.player.play()
    
    @Slot(int, str)
    def handle_lookahead_ready(self, block_idx: int, desc: str):
        desc = clean_string(desc)
        self.lookahead_data[block_idx] = desc
        self.save_lookahead_data()
        self.fetching_block = -1
        if self.waiting_for_block == block_idx:
            self.waiting_for_block = -1
            self.mw.speak(desc)
            self.mw.player.play()
        self.check_and_fetch_lookahead()

    def start_batch_lookahead(self):
        """Pre-generates every 30-second block's description up front so playback never has to pause and wait."""
        if self._batch_inflight:
            self.mw.speak("Batch processing is already running.")
            return
        if not self.mw.media_path:
            self.mw.speak("Please open a video file first.")
            return
        dur = self.mw.player.duration()
        if dur <= 0:
            self.mw.speak("Video duration is unknown. Try playing the video for a second first.")
            return

        max_blocks = (dur // 30000) + 1
        self.init_lookahead_data()
        if self.lookahead_data:
            was_playing = self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            if was_playing: self.mw.player.pause()
            dialog = ExistingDescriptionDialog(len(self.lookahead_data), self.mw)
            accepted = dialog.exec() == QDialog.DialogCode.Accepted
            if was_playing: self.mw.player.play()
            if not accepted:
                return
            if dialog.choice == "use_existing":
                self.enable_lookahead_mode()
                return
            self.lookahead_data.clear()
            self.save_lookahead_data()

        self.mw.speak(f"Starting batch processing of {max_blocks} blocks.")
        self._batch_inflight = True
        self.is_auto_describing = False
        self.auto_describe_timer.stop()
        self.lookahead_mode = False
        threading.Thread(target=self._batch_lookahead_worker, args=(max_blocks,), daemon=True).start()

    def _batch_lookahead_worker(self, max_blocks: int):
        try:
            for block_idx in range(max_blocks):
                if block_idx in self.lookahead_data and not self.lookahead_data[block_idx].startswith("Error"):
                    continue

                # describe_block reads the preceding blocks out of lookahead_data itself,
                # and each one lands there below before the next block is described.
                desc = self.describe_block(self.mw.media_path, block_idx)
                if desc and not desc.startswith("Error"):
                    self.lookahead_data[block_idx] = desc
                    self.save_lookahead_data()
        except Exception:
            pass
        finally:
            self.batchCompleteSignal.emit()

    @Slot()
    def on_batch_complete(self):
        self._batch_inflight = False
        self.mw.speak("Batch processing is complete.")
    
    @Slot(str)
    def handle_full_story_ready(self, text: str):
        if text.startswith("Error"):
            self.mw.speak(text)
            return
        self.mw.updateStatus.emit("Full video story ready.")
        self._show_story(text)

    def _show_story(self, text: str):
        """Pauses playback and opens the story in the main window's story panel; closing it
        (Escape) resumes playback if it was playing."""
        if not self.mw.story_panel.isVisible():
            self._story_was_playing = self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            if self._story_was_playing: self.mw.player.pause()
        self.mw.speak("Video story. Use the arrow keys to read, Escape to close.")
        self.mw.story_panel.show_text(text)

    def _on_story_panel_closed(self):
        if self._story_was_playing: self.mw.player.play()
        self.mw.video_widget.setFocus()

    def generate_full_video_story(self, force_regenerate: bool = False):
        """
        Narrates the whole video in one AI call using up to 100 evenly-spaced frames.
        The result is cached: embedded in the video file itself for MP4/MOV, or a legacy
        sidecar .story.txt for other containers (story_path stays None for the embedded
        case -- _full_story_worker uses that to decide where to save the result).
        """
        if not self.mw.media_path:
            self.mw.speak("Please open a video file first.")
            return

        story_path = None if self._uses_embedded_metadata() else self.mw.get_data_file_path(".story.txt")
        if not force_regenerate:
            cached_text = ""
            if story_path is None:
                cached_text = mp4_metadata.read_metadata(self.mw.media_path).get("story", "")
            elif os.path.exists(story_path):
                try:
                    with open(story_path, 'r', encoding='utf-8') as f:
                        cached_text = f.read()
                except Exception:
                    pass
            if cached_text:
                self._show_story(cached_text)
                return

        if self._inflight:
            self.mw.speak("A vision process is already running.")
            return

        self.mw.speak("Generating full video story. This may take a few moments.")
        self._inflight = True
        threading.Thread(target=self._full_story_worker, args=(story_path,), daemon=True).start()

    def _full_story_worker(self, story_path: str | None):
        import cv2 
        try:
            cap = cv2.VideoCapture(self.mw.media_path)
            if not cap.isOpened():
                self.mw.updateStatus.emit("Error: Cannot open video file.")
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            dur_ms = int((frame_count / fps) * 1000) if fps > 0 else self.mw.player.duration()

            if dur_ms <= 0:
                self.mw.updateStatus.emit("Error: Unknown video duration.")
                return

            # Frame-grid batching (Ctrl+C) packs 9 frames per image, so sample more densely.
            max_frames = 600 if self.mw.grid_frames_enabled else 100
            interval_ms = max(1000.0, dur_ms / float(max_frames))
            self.mw.updateStatus.emit(f"Extracting up to {max_frames} frames at ~{interval_ms/1000:.1f}s intervals...")
            raw_frames = []
            i = 0
            while len(raw_frames) < max_frames:
                pos_ms = float(i * interval_ms)
                if pos_ms >= dur_ms:
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
                ok, frame = cap.read()
                if ok and frame is not None:
                    raw_frames.append(frame)
                else:
                    break
                i += 1
            cap.release()

            if not raw_frames:
                self.mw.updateStatus.emit("Error: No frames extracted.")
                return
            frames_b64, grid_note = self._prepare_frames_for_api(raw_frames)

            if self._is_local_model():
                system_prompt = (
                    f"Narrate these {len(raw_frames)} frames as one story. "
                    f"Under {self.mw.current_max_words} words. Do not say 'image' or 'frame'. "
                    f"Tone: {self.mw.current_mode}."
                )
                user_text = "Narrate the video from start to end."
            else:
                system_prompt = (
                    f"You are a professional cinematic audio describer. Respond in {self.mw.current_language}. "
                    f"Limit the story to exactly {self.mw.current_max_words} words. "
                    f"I am providing you with {len(raw_frames)} visual frames extracted every {interval_ms/1000:.1f} seconds, spanning the entire video. "
                    f"Translate these sequential visual moments into a seamless, continuous, real-time story. "
                    f"CRITICAL INSTRUCTIONS: "
                    f"1. Never use words like 'image', 'frame', or 'shows'. Treat the input as a living world. "
                    f"2. Write in the immediate present tense and active voice. "
                    f"3. Weave the actions fluidly, naturally inferring the bridging movements between the gaps. "
                    f"4. Word all of this in a '{self.mw.current_mode}' tone. The tone changes how it is worded, never which events get told."
                )
                user_text = "Watch the entire sequence and narrate the unfolding events fluidly from beginning to end."
            if grid_note:
                system_prompt += " " + grid_note
            self.mw.updateStatus.emit("Sending frames to API...")
            response = self.mw.api_client.send(
                model_dict=self.mw.current_model_dict,
                frames_b64=frames_b64,
                system_prompt=system_prompt,
                user_text_blocks=[user_text],
                expect_json=False,
                language=self.mw.current_language
            )

            if response and isinstance(response, dict):
                if "choices" in response:
                    result_text = clean_string(response["choices"][0]["message"]["content"].strip())
                    if story_path is None:
                        self._update_embedded_metadata(story=result_text)
                    else:
                        try:
                            os.makedirs(os.path.dirname(story_path), exist_ok=True)
                            with open(story_path, 'w', encoding='utf-8') as f:
                                f.write(result_text)
                        except Exception:
                            pass
                    self.fullStoryReady.emit(result_text)
                elif "error" in response:
                    self.fullStoryReady.emit(f"Error: {response['error']}")
            else:
                self.fullStoryReady.emit("Error: Failed to generate full story.")

        except Exception as e:
            self.mw.updateStatus.emit(f"Error generating full story: {e}")
        finally:
            self._inflight = False