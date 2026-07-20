# Cinevoice

An accessible, keyboard-driven video player for blind and low-vision users, built with PySide6 (Qt). Cinevoice uses AI vision-language models to describe what's happening on screen through your screen reader or built-in text-to-speech, so you can follow along with any video.

## Features

- **Full keyboard control** — every action has a shortcut; see [help.html](help.html) for the complete list.
- **AI scene description** — ten built-in prompt presets, two customizable slots, and free-form questions (typed or spoken).
- **Continuous narration** — describes the video in real time as it plays, or pre-generates a full 30-second-block narration in the background.
- **Full-video cinematic story** — one AI-generated narrative summary of the entire video.
- **12 narration styles** — from neutral audio description to cinematic, poetic, humorous, or explicit.
- **Online or fully offline** — use Grok, Gemini, OpenAI, or Mistral with your own API key, or download a vision-language model (OpenVINO, 1.9–5.5 GB) and run everything locally on your own CPU/GPU with no internet connection or API key.
- **Screen-reader native** — built to work with NVDA, JAWS, and Windows Narrator.

## Download

Grab the latest build from the [Releases](../../releases) page — download the zip, extract it anywhere, and run `Cinevoice.exe`. No installer needed.

## Building from source

Requires Python 3.11+ and the dependencies in this folder (PySide6, opencv-python, openvino-genai, requests, keyring, speech_recognition, pyaudio, pywin32). Then:

```
pyinstaller Cinevoice.spec --clean
```

The built app appears in `dist/Cinevoice/`.

## Licensing

Cinevoice's own code is MIT-licensed — see [LICENSE.txt](LICENSE.txt). Third-party components it bundles are listed with their licenses in [THIRD-PARTY-LICENSES.txt](THIRD-PARTY-LICENSES.txt).

## Issues & Feedback

Found a bug or have a feature request? Please [open an issue](../../issues).
