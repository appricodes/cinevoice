"""
Player volume from 0% to 200%.

QAudioOutput clamps its volume to 1.0, so it alone can't go above 100%. A
QAudioBufferOutput is attached to the player for its whole lifetime (buffers only arrive
if it's attached before playback starts), receiving the same decoded audio in real time.
Up to 100% the native QAudioOutput plays it and the buffers are ignored; above 100% the
native output is muted and the buffers are amplified and played through a QAudioSink.
"""
import numpy as np
from PySide6.QtCore import QObject
from PySide6.QtMultimedia import QAudioBufferOutput, QAudioFormat, QAudioOutput, QAudioSink, QMediaDevices, QMediaPlayer

MAX_VOLUME = 2.0
# Kept short: this is the extra delay of the boosted audio relative to the video.
SINK_BUFFER_MS = 200


class VolumeBooster(QObject):
    def __init__(self, player: QMediaPlayer, audio_output: QAudioOutput, parent=None):
        super().__init__(parent)
        self._player = player
        self._audio_output = audio_output
        self._volume = 1.0
        self._sink = None
        self._sink_format = None
        self._sink_dev = None

        self._buffer_output = QAudioBufferOutput(self)
        self._buffer_output.audioBufferReceived.connect(self._on_buffer)
        player.setAudioBufferOutput(self._buffer_output)
        player.playbackStateChanged.connect(self._on_playback_state)
        self.set_volume(1.0)

    def volume(self) -> float:
        return self._volume

    def set_volume(self, volume: float) -> float:
        self._volume = max(0.0, min(MAX_VOLUME, round(volume, 2)))
        if self._volume <= 1.0:
            self._audio_output.setVolume(self._volume)
            self._stop_sink()
        else:
            self._audio_output.setVolume(0.0)
        return self._volume

    def _stop_sink(self):
        # Drops whatever is still queued, so pausing/seeking/leaving boost is immediate.
        if self._sink is not None:
            self._sink.stop()
        self._sink_dev = None

    def _on_playback_state(self, state):
        if state != QMediaPlayer.PlaybackState.PlayingState:
            self._stop_sink()

    def _on_buffer(self, buffer):
        if self._volume <= 1.0 or not buffer.isValid():
            return
        fmt = buffer.format()
        if self._sink is None or fmt != self._sink_format:
            self._stop_sink()
            self._sink = QAudioSink(QMediaDevices.defaultAudioOutput(), fmt, self)
            self._sink.setBufferSize(fmt.bytesForDuration(SINK_BUFFER_MS * 1000))
            self._sink_format = fmt
        if self._sink_dev is None:
            self._sink_dev = self._sink.start()
            if self._sink_dev is None:
                return
        self._sink_dev.write(self._amplify(bytes(buffer.constData()), fmt.sampleFormat()))

    def _amplify(self, data: bytes, sample_format) -> bytes:
        gain = self._volume
        if sample_format == QAudioFormat.SampleFormat.Float:
            samples = np.frombuffer(data, dtype=np.float32) * gain
            return np.clip(samples, -1.0, 1.0).astype(np.float32).tobytes()
        if sample_format == QAudioFormat.SampleFormat.Int16:
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) * gain
            return np.clip(samples, -32768, 32767).astype(np.int16).tobytes()
        if sample_format == QAudioFormat.SampleFormat.Int32:
            samples = np.frombuffer(data, dtype=np.int32).astype(np.float64) * gain
            return np.clip(samples, -2147483648, 2147483647).astype(np.int32).tobytes()
        if sample_format == QAudioFormat.SampleFormat.UInt8:
            samples = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) * gain + 128.0
            return np.clip(samples, 0, 255).astype(np.uint8).tobytes()
        return data
