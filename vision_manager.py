"""
Captures video frames and sends them to the AI model for description, plus the
supporting features built on top of that: scene-change detection (k-means color
clustering), continuous 30-second narration, batch pre-generation, and full-video
story summaries.
"""
import os
import json
import threading
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QDialog
from PySide6.QtMultimedia import QMediaPlayer

from config import MAX_IMAGE_DIM, SCENE_KMEAN_IMAGE_SIZE, SCENE_FRAME_INTERVAL, SCENE_KMEAN_CLUSTERS
from utils import clean_string, encode_and_resize_frame
from ui_components import CustomPromptDialog

class VisionManager(QObject):
    """Owns all frame-capture and AI-description logic; one instance per open video."""
    grokResponseReady = Signal(object)  # named after the original provider; now fires for any model
    lookaheadBlockReady = Signal(int, str)
    batchCompleteSignal = Signal()
    fullStoryReady = Signal(str)
    
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.chat_history = []
        self._history_lock = threading.Lock()
        self._inflight = False
        self._batch_inflight = False
        
        self.scene_timestamps = []
        
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
        
        self.grokResponseReady.connect(self.handle_grok_response)
        self.lookaheadBlockReady.connect(self.handle_lookahead_ready)
        self.batchCompleteSignal.connect(self.on_batch_complete)
        self.fullStoryReady.connect(self.handle_full_story_ready)
    
    def assign_custom_prompt(self, slot):
        was_playing = self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if was_playing: self.mw.player.pause()
        
        dialog = CustomPromptDialog(self.mw)
        dialog.setWindowTitle(f"Assign Custom Prompt F{slot}")
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if was_playing: self.mw.player.play()
            prompt_text, frames_count, frames_interval, max_words = dialog.get_data()
            prompt_data = {"title": f"Custom {slot - 6}", "prompt": prompt_text, "frames_count": frames_count,
                           "frames_interval": frames_interval, "max_words": max_words}
            
            if slot == 7:
                self.mw.custom_prompt_f7 = prompt_data
            elif slot == 8:
                self.mw.custom_prompt_f8 = prompt_data
            self.mw.current_prompt_data = prompt_data
            
            self.mw.speak(f"Assigned to F{slot}. Processing...")
            cur_ms = self.mw.player.position()
            if cur_ms is not None and cur_ms >= 0:
                threading.Thread(target=self.execute_prompt, args=(cur_ms, self.mw.current_prompt_data), daemon=True).start()
        else:
            if was_playing: self.mw.player.play()
            self.mw.video_widget.setFocus()
    
    def show_custom_prompt_dialog(self):
        was_playing = self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if was_playing: self.mw.player.pause()
        
        dialog = CustomPromptDialog(self.mw)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if was_playing: self.mw.player.play()
            prompt_text, frames_count, frames_interval, max_words = dialog.get_data()
            self.mw.current_prompt_data = {"title": "custom", "prompt": prompt_text, "frames_count": frames_count,
                                           "frames_interval": frames_interval, "max_words": max_words}
            cur_ms = self.mw.player.position()
            threading.Thread(target=self.execute_prompt, args=(cur_ms, self.mw.current_prompt_data), daemon=True).start()
        else:
            if was_playing: self.mw.player.play()
            self.mw.video_widget.setFocus()
    
    def trigger_current_prompt(self):
        self.mw.speak(self.mw.current_prompt_data.get('title', 'Processing'))
        self.is_auto_describing = False
        with self._history_lock: self.chat_history.clear()
        cur_ms = self.mw.player.position()
        if cur_ms is not None and cur_ms >= 0:
            threading.Thread(target=self.execute_prompt, args=(cur_ms, self.mw.current_prompt_data), daemon=True).start()
    
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
            captured_frames_b64 = []
            try:
                for ms_time in capture_times_ms:
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(ms_time))
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        captured_frames_b64.append(encode_and_resize_frame(frame, MAX_IMAGE_DIM, quality=90))
            finally:
                cap.release()
            
            final_user_prompt = prompt_text
            
            system_prompt_string = (
                f"You are a professional cinematic audio describer. Respond in {self.mw.current_language} using a '{self.mw.current_mode}' style. "
                f"Limit to {max_words} words. "
                f"CRITICAL INSTRUCTIONS: "
                f"1. Never use words like 'image', 'frame', 'picture', or 'shows'. Treat the visual input as a living, unfolding world. "
                f"2. Write in the immediate present tense and active voice. "
                f"3. Strictly describe ONLY what is actually, physically present in the scene. Do not hallucinate or assume unseen events."
            )
            
            with self._history_lock:
                self.chat_history.append({"role": "user", "content": final_user_prompt})
            
            self.mw.updateStatus.emit(f"Sending frames to API...")
            response = self.mw.api_client.send(
                model_dict=self.mw.current_model_dict,
                frames_b64=captured_frames_b64,
                system_prompt=system_prompt_string,
                user_text_blocks=[final_user_prompt],
                expect_json=False,
                history=self.chat_history,
                language=self.mw.current_language
            )
            self.grokResponseReady.emit(response)
        except Exception as e:
            self.grokResponseReady.emit({"error": f"Execution failed: {e}"})
    
    @Slot()
    def on_auto_describe_tick(self):
        """Runs every 5 seconds while auto-describe is on, asking only what changed since last time."""
        if not self.is_auto_describing or self.mw.player.playbackState() != QMediaPlayer.PlaybackState.PlayingState or self._inflight: return
        cur_ms = self.mw.player.position()
        if cur_ms is None or cur_ms < 5000: return
        auto_prompt_data = {"title": "Auto-Describe-Diff", "prompt": "Describe only new changes in the scene.",
                            "frames_count": 2, "frames_interval": 5.0}
        self._inflight = True
        def run():
            try:
                self.execute_prompt(cur_ms, auto_prompt_data)
            finally:
                self._inflight = False
        threading.Thread(target=run, daemon=True).start()
    
    @Slot(object)
    def handle_grok_response(self, response):
        """Speaks the AI's reply (or the error) and records it in chat_history for follow-up questions."""
        desc = "Error: Unknown response failure."
        if response and isinstance(response, dict):
            if "choices" in response:
                desc = response["choices"][0]["message"]["content"].strip()
            elif "error" in response:
                desc = response["error"]
        self.mw.speak(clean_string(desc))
        with self._history_lock:
            self.chat_history.append({"role": "assistant", "content": desc})
    
    def init_lookahead_data(self):
        """Loads any 30-second block descriptions already cached on disk for this video."""
        if not self.mw.media_path: return
        self.lookahead_json_path = self.mw.get_data_file_path(".json")
        self.lookahead_data.clear()
        if os.path.exists(self.lookahead_json_path):
            try:
                with open(self.lookahead_json_path, 'r', encoding='utf-8') as f:
                    for item in json.load(f):
                        self.lookahead_data[item.get("time", 0) // 30] = item.get("description", "")
            except Exception:
                pass
    
    def save_lookahead_data(self):
        """Writes the 30-second block descriptions to disk so they persist across sessions."""
        if not self.lookahead_json_path: return
        data = [{"time": idx * 30, "description": desc} for idx, desc in sorted(self.lookahead_data.items())]
        try:
            with open(self.lookahead_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def toggle_lookahead_mode(self):
        self.lookahead_mode = not self.lookahead_mode
        if self.lookahead_mode:
            self.mw.speak("Continuous 30-second narration enabled.")
            self.is_auto_describing = False
            self.auto_describe_timer.stop()
            self.init_lookahead_data()
            self.current_playing_block = -1
            self.check_and_fetch_lookahead()
        else:
            self.mw.speak("Continuous narration disabled.")
            self.waiting_for_block = -1
            self.fetching_block = -1
            if self.mw.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
                self.mw.player.play()
    
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
    
    def describe_block(self, media_path: str, block_idx: int, prev_desc: str = None) -> str:
        """
        Describes the 30-second block at block_idx using 3 sampled frames. prev_desc, the
        previous block's description, is included as context so consecutive blocks read as
        one continuous story rather than restarting cold each time.
        """
        import cv2
        start_ms = block_idx * 30000
        self.mw.updateStatus.emit(f"Extracting frames for 30s block {block_idx}...")
        frames_b64 = []
        cap = cv2.VideoCapture(media_path)
        if not cap.isOpened():
            return f"Error: Cannot open video file {os.path.basename(media_path)}"
        try:
            for i in range(3):
                cap.set(cv2.CAP_PROP_POS_MSEC, float(start_ms + (i * 10000)))
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames_b64.append(encode_and_resize_frame(frame, MAX_IMAGE_DIM))
                else:
                    break
        finally:
            cap.release()
        
        if not frames_b64: return "Error: No frames extracted."
        
        system_prompt = (
            f"You are a professional cinematic audio describer. Respond in {self.mw.current_language}. "
            f"Limit to {self.mw.current_max_words} words. "
            f"Translate these sequential visual moments into a seamless, continuous real-time story. "
            f"CRITICAL INSTRUCTIONS: "
            f"1. Never use words like 'image', 'frame', or 'shows'. Treat the input as a living world. "
            f"2. Write in the immediate present tense and active voice. "
            f"3. Weave the actions fluidly. "
            f"4. Focus on exact physical actions, expressions, and spatial movements."
        )
        user_text = "Narrate the unfolding events fluidly."
        if prev_desc:
            user_text += f"\n\nCONTEXT FROM PRECEDING SCENE:\n\"\"\"\n{prev_desc}\n\"\"\"\nContinue seamlessly without summarizing."
        
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
        prev_desc = self.lookahead_data.get(block_idx - 1, "")
        desc = self.describe_block(self.mw.media_path, block_idx, prev_desc)
        if desc.startswith("Error"):
            self.fetching_block = -1
        else:
            self.lookaheadBlockReady.emit(block_idx, desc)
    
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
        self.mw.speak(f"Starting batch processing of {max_blocks} blocks.")
        self.init_lookahead_data()
        self._batch_inflight = True
        self.is_auto_describing = False
        self.auto_describe_timer.stop()
        self.lookahead_mode = False
        threading.Thread(target=self._batch_lookahead_worker, args=(max_blocks,), daemon=True).start()

    def _batch_lookahead_worker(self, max_blocks: int):
        try:
            prev_desc = ""
            for block_idx in range(max_blocks):
                if block_idx in self.lookahead_data and not self.lookahead_data[block_idx].startswith("Error"):
                    prev_desc = self.lookahead_data[block_idx]
                    continue

                desc = self.describe_block(self.mw.media_path, block_idx, prev_desc)
                if desc and not desc.startswith("Error"):
                    self.lookahead_data[block_idx] = desc
                    self.save_lookahead_data()
                    prev_desc = desc
        except Exception:
            pass
        finally:
            self.batchCompleteSignal.emit()

    @Slot()
    def on_batch_complete(self):
        self._batch_inflight = False
        self.mw.speak("Batch processing is complete.")
    
    def _get_frame_clusters(self, frame, k: int):
        """Returns a frame's k dominant colors and each one's share of the pixels, for scene-cut comparison."""
        import cv2
        import numpy as np
        small = cv2.resize(frame, SCENE_KMEAN_IMAGE_SIZE)
        Z = small.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        weights = np.zeros(len(centers), dtype=np.float32)
        total_pixels = Z.shape[0]
        for i in range(len(centers)):
            weights[i] = np.sum(labels == i) / total_pixels
        return centers, weights

    def _compare_clusters(self, centers1, weights1, centers2, weights2) -> float:
        """Distance between two frames' color palettes: higher means a bigger visual change (likely a cut)."""
        import numpy as np
        total_diff = 0.0
        for i, c1 in enumerate(centers1):
            dists = np.linalg.norm(centers2 - c1, axis=1)
            total_diff += np.min(dists) * weights1[i]
        for j, c2 in enumerate(centers2):
            dists = np.linalg.norm(centers1 - c2, axis=1)
            total_diff += np.min(dists) * weights2[j]
        return float(total_diff)
        
    def analyze_video_scenes(self):
        """Loads cached scene-change timestamps for this video, or starts detecting them in the background."""
        if not self.mw.media_path: return
        
        self.scene_timestamps = []
        cache_path = self.mw.get_data_file_path("_scenes_time.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.scene_timestamps = json.load(f)
                return
            except Exception:
                pass
                
        threading.Thread(target=self._scene_analysis_worker, args=(cache_path,), daemon=True).start()

    def _scene_analysis_worker(self, cache_path: str):
        """
        Samples a frame every SCENE_FRAME_INTERVAL seconds and measures the color-palette
        change since the previous sample. A timestamp counts as a scene cut only if its
        change is more than 2 standard deviations above the video's average change, which
        filters out ordinary motion and keeps only genuine cuts.
        """
        import cv2
        cap = cv2.VideoCapture(self.mw.media_path)
        if not cap.isOpened(): return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps

        diffs_data = []
        
        ret, frame = cap.read()
        if not ret: 
            cap.release()
            return
            
        prev_centers, prev_weights = self._get_frame_clusters(frame, SCENE_KMEAN_CLUSTERS)
        current_sec = SCENE_FRAME_INTERVAL

        while current_sec <= duration_sec:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_sec * 1000)
            ret, frame = cap.read()
            if not ret: break
            
            centers, weights = self._get_frame_clusters(frame, SCENE_KMEAN_CLUSTERS)
            diff = self._compare_clusters(prev_centers, prev_weights, centers, weights)
            
            diffs_data.append((int(current_sec * 1000), diff))
            prev_centers, prev_weights = centers, weights
            current_sec += SCENE_FRAME_INTERVAL

        cap.release()

        if not diffs_data: return

        diff_values = [d[1] for d in diffs_data]
        mean_diff = sum(diff_values) / len(diff_values)
        variance = sum([((x - mean_diff) ** 2) for x in diff_values]) / len(diff_values)
        std_dev = variance ** 0.5
        threshold = mean_diff + (2 * std_dev)

        self.scene_timestamps = [d[0] for d in diffs_data if d[1] > threshold]
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(self.scene_timestamps, f)
        except Exception:
            pass
        
    def seek_to_scene(self, direction_forward: bool):
        """Jumps playback to the next or previous timestamp in scene_timestamps."""
        if not self.mw.media_path: return
        
        if not self.scene_timestamps:
            self.mw.speak("Scene analysis is still calculating in the background.")
            return
            
        cur_ms = self.mw.player.position()
        buffer_ms = SCENE_FRAME_INTERVAL * 1000 
        target_ms = -1
        
        if direction_forward:
            for t in self.scene_timestamps:
                if t > cur_ms + buffer_ms:
                    target_ms = t
                    break
        else:
            for t in reversed(self.scene_timestamps):
                if t < cur_ms - buffer_ms:
                    target_ms = t
                    break
                    
        if target_ms != -1:
            self.mw.sceneFoundReady.emit(target_ms)
        else:
            self.mw.speak("No further scenes available.")

    @Slot(str)
    def handle_full_story_ready(self, text: str):
        self.mw.updateStatus.emit("Full video story ready.")
        self.mw.speak(text)

    def generate_full_video_story(self, force_regenerate: bool = False):
        """Narrates the whole video in one AI call using up to 100 evenly-spaced frames; result is cached to disk."""
        if not self.mw.media_path:
            self.mw.speak("Please open a video file first.")
            return

        story_path = self.mw.get_data_file_path(".story.txt")
        if not force_regenerate and os.path.exists(story_path):
            self.mw.speak("Reading cached story.")
            try:
                with open(story_path, 'r', encoding='utf-8') as f:
                    cached_text = f.read()
                self.mw.speak(cached_text)
                return
            except Exception:
                pass

        if self._inflight:
            self.mw.speak("A vision process is already running.")
            return

        self.mw.speak("Generating full video story. This may take a few moments.")
        self._inflight = True
        threading.Thread(target=self._full_story_worker, args=(story_path,), daemon=True).start()

    def _full_story_worker(self, story_path: str):
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

            interval_ms = max(3000.0, dur_ms / 100.0)
            self.mw.updateStatus.emit(f"Extracting up to 100 frames at ~{interval_ms/1000:.1f}s intervals...")
            frames_b64 = []
            i = 0
            while len(frames_b64) < 100:
                pos_ms = float(i * interval_ms)
                if pos_ms >= dur_ms:
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames_b64.append(encode_and_resize_frame(frame, MAX_IMAGE_DIM))
                else:
                    break
                i += 1
            cap.release()

            if not frames_b64:
                self.mw.updateStatus.emit("Error: No frames extracted.")
                return

            system_prompt = (
                f"You are a professional cinematic audio describer. Respond in {self.mw.current_language}. "
                f"Use a '{self.mw.current_mode}' writing style. Limit the story to exactly {self.mw.current_max_words} words. "
                f"I am providing you with {len(frames_b64)} visual frames extracted every {interval_ms/1000:.1f} seconds, spanning the entire video. "
                f"Translate these sequential visual moments into a seamless, continuous, real-time story. "
                f"CRITICAL INSTRUCTIONS: "
                f"1. Never use words like 'image', 'frame', or 'shows'. Treat the input as a living world. "
                f"2. Write in the immediate present tense and active voice. "
                f"3. Weave the actions fluidly, naturally inferring the bridging movements between the gaps."
            )

            user_text = "Watch the entire sequence and narrate the unfolding events fluidly from beginning to end."
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
                    try:
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