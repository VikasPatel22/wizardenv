# 🧙 WizardAI SDK

<p align="center">
  <img src="https://img.shields.io/badge/version-2.0.0-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/python-3.9%2B-brightgreen?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/endpoint-any%20OpenAI--compatible-purple?style=for-the-badge" />
</p>

> **A powerful, all-in-one Python SDK for AI integration** — combining conversational AI, computer vision, speech I/O, memory management, and a flexible plugin system into a single, easy-to-use module.  
> Works with **any OpenAI-compatible endpoint** — cloud or local.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Modules](#modules)
  - [WizardAI Core](#wizardai-core)
  - [AIClient](#aiclient)
  - [ConversationAgent](#conversationagent)
  - [MemoryManager](#memorymanager)
  - [VisionModule](#visionmodule)
  - [SpeechModule](#speechmodule)
  - [Plugin System](#plugin-system)
  - [Exceptions](#exceptions)
- [Folder Structure](#folder-structure)
- [Configuration Reference](#configuration-reference)
- [Publishing to PyPI](#publishing-to-pypi)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Description |
|---|---|
| 🌐 **Any endpoint** | Works with OpenAI, Anthropic, Ollama, LM Studio, vLLM, or any OpenAI-compatible REST API |
| 🌊 **Streaming** | First-class SSE streaming via `chat_stream()` / `complete_stream()` |
| 💬 **Conversation Agent** | AIML-style pattern matching with wildcards, priorities, and context |
| 🧠 **Memory Manager** | Short-term (sliding window) + long-term (key-value) + JSON persistence |
| 👁️ **Vision Module** | Real-time webcam capture, face detection, and frame streaming via OpenCV |
| 🎙️ **Speech Module** | STT (Google / Sphinx / Whisper) + TTS (pyttsx3 / gTTS / ElevenLabs) |
| 🔌 **Plugin System** | Register, load, and dispatch custom skills from files or directories |
| 🔁 **Auto-retry** | Exponential back-off on transient API errors, built-in rate limiting |
| 🖥️ **Interactive REPL** | Built-in terminal chat loop with optional voice mode |

---

## Installation

### Minimal (core only)

```bash
pip install wizardai
# or
pip install -r requirements.txt
```

### With specific features

```bash
# Computer vision
pip install "wizardai[vision]"

# Speech (STT + TTS)
pip install "wizardai[speech]"

# High-quality offline STT (Whisper)
pip install "wizardai[whisper]"

# Everything
pip install "wizardai[all]"
```

### From source

```bash
git clone https://github.com/yourusername/wizardai-sdk.git
cd wizardai-sdk
pip install -e ".[all]"
```

---

## Quick Start

```python
import wizardai

# Point at any OpenAI-compatible endpoint
with wizardai.WizardAI(
    endpoint="https://api.openai.com/v1/chat/completions",
    api_key="sk-...",
    default_model="gpt-4o-mini",
) as wiz:

    # Pattern-based chat (no API call)
    wiz.agent.add_pattern("hello *", "Hello there, {wildcard}!")
    print(wiz.chat("hello world"))           # → "Hello there, world!"

    # LLM call
    print(wiz.ask("What is the speed of light?"))

    # Streaming LLM call
    for chunk in wiz.ask_stream("Write me a haiku about Python."):
        print(chunk, end="", flush=True)

    # Long-term memory
    wiz.remember("user_name", "Alice")
    print(wiz.recall("user_name"))           # → "Alice"
```

---

## Modules

### WizardAI Core

`WizardAI` is the top-level orchestrator. It wires all sub-modules together and exposes convenience shortcuts for the most common tasks.

```python
from wizardai import WizardAI

wiz = WizardAI(
    endpoint="https://api.openai.com/v1/chat/completions",  # any OpenAI-compatible URL
    api_key="sk-...",                  # or set WIZARDAI_API_KEY env var
    default_model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.7,
    enable_vision=True,           # open webcam on start()
    enable_speech=True,           # init STT/TTS on start()
    stt_backend="google",
    tts_backend="pyttsx3",
    agent_name="WizardBot",
    fallback_response="I'm not sure about that.",
    max_history=50,
    memory_path="session.json",   # persist memory to disk
    system_prompt="You are a helpful assistant.",
    log_level="INFO",
    data_dir="./wizardai_data",
)

wiz.start()   # opens camera, initialises speech engine, notifies plugins

# --- Chat shortcuts ---
reply = wiz.chat("hello")              # tries patterns first, then LLM
reply = wiz.ask("Tell me a joke")      # always calls the LLM (non-streaming)

# --- Streaming shortcuts ---
for chunk in wiz.ask_stream("Write a poem"):
    print(chunk, end="", flush=True)

# --- Voice shortcuts ---
wiz.say("Hello from WizardAI!")        # TTS
text = wiz.listen(timeout=5)           # STT
reply = wiz.voice_chat()               # listen → chat → say

# --- Vision shortcuts ---
frame = wiz.capture()                  # single frame (numpy ndarray)
path  = wiz.snapshot("photo.jpg")      # capture + save

# --- Memory shortcuts ---
wiz.remember("city", "Paris")
city = wiz.recall("city")             # → "Paris"
history = wiz.get_history(n=5)        # last 5 turns as list of dicts

wiz.stop()

# Use as a context manager
with WizardAI(endpoint="...", api_key="sk-...") as wiz:
    print(wiz.ask("Hi!"))

# Launch interactive terminal REPL
wiz.run_repl()
wiz.run_repl(voice_mode=True)          # voice I/O
```

---

### AIClient

Lightweight client for any OpenAI-compatible REST endpoint. You supply the URL and model; WizardAI handles retries, rate limiting, and streaming.

**Supported endpoints (examples)**

| Provider | Endpoint URL | Notes |
|---|---|---|
| OpenAI | `https://api.openai.com/v1/chat/completions` | Requires `OPENAI_API_KEY` |
| Anthropic (proxy) | Your proxy URL | Anthropic's native API is not OpenAI-compatible |
| Ollama (local) | `http://localhost:11434/v1/chat/completions` | No key needed |
| LM Studio (local) | `http://localhost:1234/v1/chat/completions` | No key needed |
| vLLM | `http://localhost:8000/v1/chat/completions` | Configurable |
| Together AI | `https://api.together.xyz/v1/chat/completions` | Requires API key |
| Groq | `https://api.groq.com/openai/v1/chat/completions` | Requires API key |
| Azure OpenAI | Your Azure deployment URL | Requires Azure key |
| Any custom REST | Your URL | Must be OpenAI-compatible |

```python
from wizardai import AIClient

# --- Any OpenAI-compatible endpoint ---
client = AIClient(
    endpoint="https://api.openai.com/v1/chat/completions",
    api_key="sk-...",          # or set WIZARDAI_API_KEY env var
    model="gpt-4o-mini",
)

# Single-turn completion (non-streaming)
resp = client.complete("Write a haiku about Python.")
print(resp.text)
print(resp.usage)              # {"prompt_tokens": …, "completion_tokens": …}
print(resp.latency_ms)         # round-trip time in ms

# Multi-turn chat (non-streaming)
messages = [
    {"role": "user",      "content": "My name is Bob."},
    {"role": "assistant", "content": "Nice to meet you, Bob!"},
    {"role": "user",      "content": "What is my name?"},
]
resp = client.chat(messages, system_prompt="You are a helpful assistant.")
print(resp.text)               # → "Your name is Bob."

# --- Streaming ---
for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)

for chunk in client.complete_stream("Explain recursion step by step."):
    print(chunk, end="", flush=True)

# --- Local / self-hosted (no API key required) ---
local_client = AIClient(
    endpoint="http://localhost:11434/v1/chat/completions",
    model="llama3",
)
resp = local_client.complete("Hello!")

# --- Runtime config changes ---
client.set_model("gpt-4o")
client.set_api_key("sk-new-key...")
client.set_endpoint("https://api.groq.com/openai/v1/chat/completions")
```

**Using environment variables**

```bash
export WIZARDAI_API_KEY="sk-..."
```

```python
# api_key is resolved automatically from WIZARDAI_API_KEY
client = AIClient(
    endpoint="https://api.openai.com/v1/chat/completions",
    model="gpt-4o",
)
```

---

### ConversationAgent

AIML-style rule-based chat engine with wildcard patterns, priorities, context rules, and callable templates — with memory integration.

```python
from wizardai import ConversationAgent, Pattern, MemoryManager

mem   = MemoryManager()
agent = ConversationAgent(name="Aria", fallback="I don't know!", memory=mem)

# --- Simple patterns ---
agent.add_pattern("hello", "Hi there!")
agent.add_pattern("what is your name", "I'm Aria, your AI assistant.")

# --- Wildcards ---
#   *   matches any sequence of words → {wildcard}
#   ?   matches exactly one word
agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")
agent.add_pattern("what is ? plus ?", "Let me calculate that for you.")

# --- Callable template (dynamic response) ---
import random
agent.add_pattern(
    "tell me a joke",
    lambda text, ctx: random.choice([
        "Why do programmers prefer dark mode? Because light attracts bugs!",
        "I told a joke about UDP once. I don't know if you got it.",
    ]),
)

# --- Priority (higher wins when patterns overlap) ---
agent.add_pattern("hello world", "Special hello-world greeting!", priority=10)

# --- Context-aware patterns ---
agent.add_pattern("yes", "Great, let's proceed!",  context="confirm_action")
agent.add_pattern("no",  "OK, I'll cancel that.",  context="confirm_action")
agent.set_context("confirm_action")

# --- Pattern object (full control) ---
agent.add_pattern_obj(Pattern(
    pattern="weather in *",
    template="Checking weather for {wildcard}…",
    priority=5,
    tags=["weather", "location"],
))

# Chat
print(agent.respond("hello"))               # → "Hi there!"
print(agent.respond("my name is Charlie"))  # → "Nice to meet you, Charlie!"
print(agent.respond("tell me a joke"))      # → random joke

# Introspection
agent.list_patterns()
agent.remove_pattern("hello")
agent.clear_patterns()
```

---

### MemoryManager

Provides a sliding-window conversation history (short-term) plus a persistent key-value store (long-term), with optional JSON disk persistence.

```python
from wizardai import MemoryManager

mem = MemoryManager(
    max_history=20,
    persist_path="mem.json",
)

# --- Short-term (conversation history) ---
mem.add_message("user",      "What's the capital of France?")
mem.add_message("assistant", "Paris!")

history = mem.get_history()
history = mem.get_history(n=5)
history = mem.get_history(role_filter="user")

api_msgs = mem.get_messages_for_api()     # [{"role": …, "content": …}, …]

last = mem.last_message()
results = mem.search_history("France", top_k=3)
mem.clear_history()

# --- Long-term memory ---
mem.remember("user_name", "Alice")
mem.remember("preferences", {"theme": "dark", "lang": "en"})

name  = mem.recall("user_name")
prefs = mem.recall("preferences")
mem.forget("user_name")
keys  = mem.list_memories()

# --- Ephemeral session context (not persisted) ---
mem.set_context("current_topic", "weather")
topic = mem.get_context("current_topic")
mem.clear_context()

# --- Persistence ---
mem.save("backup.json")
mem.load("backup.json")
```

---

### VisionModule

Real-time webcam capture and image processing powered by OpenCV.

```python
from wizardai import VisionModule

cam = VisionModule(device_id=0, width=1280, height=720, fps=30)
cam.open()

frame = cam.capture_frame()
cam.save_frame(frame, "snapshot.jpg")
b64 = cam.encode_to_base64(frame)

faces = cam.detect_faces(frame)

def on_frame(frame):
    faces = cam.detect_faces(frame)
    if faces:
        print(f"Detected {len(faces)} face(s)")

cam.start_stream(callback=on_frame, show_preview=True)
import time; time.sleep(10)
cam.stop_stream()

cam.close()
```

---

### SpeechModule

Speech recognition (STT) and text-to-speech (TTS) with multiple backend options.

**STT backends:** `google` (online), `sphinx` (offline), `whisper` (offline)  
**TTS backends:** `pyttsx3` (offline), `gtts` (online), `elevenlabs` (online)

```python
from wizardai import SpeechModule

speech = SpeechModule(
    stt_backend="google",
    tts_backend="pyttsx3",
    language="en-US",
    tts_rate=150,
)
speech.init_tts()

speech.say("Hello, I am WizardAI!")
text = speech.listen(timeout=5.0)
text = speech.transcribe_file("audio.wav")

for word in speech.stream_say("Generating token by token…"):
    print(word, end=" ", flush=True)
```

---

### Plugin System

Extend WizardAI with custom skills by subclassing `PluginBase` and registering with `PluginManager`.

```python
from wizardai import PluginBase
from typing import Optional

class WeatherPlugin(PluginBase):
    name        = "weather"
    description = "Returns mock weather data for any city."
    version     = "1.0.0"
    triggers    = ["weather in *", "what's the weather in *"]

    def on_message(self, text: str, context: dict) -> Optional[str]:
        city = text.split("in", 1)[-1].strip()
        return f"The weather in {city} is sunny, 25 °C."
```

```python
from wizardai import PluginManager

manager = PluginManager()
manager.register(WeatherPlugin, config={"api_key": "abc123"})

response = manager.dispatch("weather in Paris", context={})
print(response)

manager.load_from_file("my_plugin.py")
manager.load_from_directory("./plugins/")
manager.start_all()
manager.stop_all()
```

---

### Exceptions

All exceptions inherit from `WizardAIError`.

```python
from wizardai import WizardAIError, APIError, VisionError, SpeechError
from wizardai.exceptions import RateLimitError, AuthenticationError

try:
    reply = wiz.ask("Hello!")
except AuthenticationError as e:
    print("Bad API key:", e.endpoint)
except RateLimitError as e:
    print("Slow down! Retry after:", e.retry_after)
except APIError as e:
    print(f"API error {e.code}:", e.message)
except WizardAIError as e:
    print("General WizardAI error:", e)
```

---

## Folder Structure

```
wizardai-sdk/
│
├── wizardai/
│   ├── __init__.py
│   ├── core.py
│   ├── ai_client.py          # ← unified custom-endpoint client
│   ├── conversation.py
│   ├── memory.py
│   ├── vision.py
│   ├── speech.py
│   ├── plugins.py
│   ├── utils.py
│   └── exceptions.py
│
├── examples/
│   └── full_demo.py
│
├── plugins/
│   └── sample_plugin.py
│
├── tests/
│   ├── __init__.py
│   ├── test_ai_client.py
│   ├── test_conversation.py
│   ├── test_memory.py
│   ├── test_plugins.py
│   └── test_core.py
│
├── .github/workflows/ci.yml
├── .gitignore
├── LICENSE
├── README.md
├── pyproject.toml
├── setup.py
├── requirements.txt
└── requirements-full.txt
```

---

## Configuration Reference

### WizardAI constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | **required** | Full URL of any OpenAI-compatible chat completions endpoint |
| `api_key` | `str` | `None` | Bearer token. Falls back to `WIZARDAI_API_KEY` env var |
| `default_model` | `str` | `"gpt-4o-mini"` | Model identifier sent with every request |
| `max_tokens` | `int` | `1024` | Max tokens per LLM response |
| `temperature` | `float` | `0.7` | Sampling temperature |
| `enable_vision` | `bool` | `False` | Open webcam on `start()` |
| `camera_device` | `int` | `0` | OpenCV camera index |
| `camera_width` | `int` | `640` | Capture width (px) |
| `camera_height` | `int` | `480` | Capture height (px) |
| `enable_speech` | `bool` | `False` | Init STT/TTS on `start()` |
| `stt_backend` | `str` | `"google"` | STT backend |
| `tts_backend` | `str` | `"pyttsx3"` | TTS engine |
| `language` | `str` | `"en-US"` | BCP-47 language code |
| `agent_name` | `str` | `"WizardBot"` | Agent display name |
| `fallback_response` | `str` | `"I'm not sure…"` | Default when no pattern matches |
| `max_history` | `int` | `50` | Short-term memory window |
| `memory_path` | `str` | `None` | Path for memory persistence |
| `system_prompt` | `str` | `None` | Default LLM system prompt |
| `log_level` | `str` | `"INFO"` | `DEBUG \| INFO \| WARNING \| ERROR` |
| `log_file` | `str` | `None` | Optional log file path |
| `data_dir` | `str` | `"./wizardai_data"` | Working data directory |

### AIClient constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `endpoint` | `str` | **required** | Full chat completions URL |
| `api_key` | `str` | `None` | Bearer token (env: `WIZARDAI_API_KEY`) |
| `model` | `str` | `"gpt-4o-mini"` | Default model identifier |
| `max_retries` | `int` | `3` | Retry attempts on transient errors |
| `retry_delay` | `float` | `1.0` | Initial retry delay in seconds (doubles each attempt) |
| `timeout` | `float` | `30.0` | HTTP request timeout in seconds |
| `rate_limit_calls` | `int` | `60` | Max calls per rate window |
| `rate_limit_period` | `float` | `60.0` | Rate-limit window in seconds |

### Environment variables

```bash
export WIZARDAI_API_KEY="your-api-key-here"
```

---

## Publishing to PyPI

Yes — you can upload WizardAI to PyPI! Here is a complete step-by-step guide.

### 1. Prepare `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "wizardai"           # must be unique on PyPI
version = "2.0.0"
description = "All-in-one Python SDK for AI integration — works with any OpenAI-compatible endpoint"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.9"
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["ai", "llm", "openai", "chatbot", "nlp"]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
dependencies = [
    "requests>=2.28",
]

[project.optional-dependencies]
vision  = ["opencv-python>=4.8"]
speech  = ["SpeechRecognition>=3.10", "pyttsx3>=2.90", "gtts>=2.3"]
whisper = ["openai-whisper>=20230918", "numpy>=1.24"]
dev     = ["pytest>=7", "black", "isort", "build", "twine"]
all     = ["wizardai[vision,speech,whisper]"]

[project.urls]
Homepage   = "https://github.com/yourusername/wizardai-sdk"
Repository = "https://github.com/yourusername/wizardai-sdk"
```

### 2. Install build tools

```bash
pip install build twine
```

### 3. Build the distribution

```bash
python -m build
# Creates:
#   dist/wizardai-2.0.0.tar.gz
#   dist/wizardai-2.0.0-py3-none-any.whl
```

### 4. Create a PyPI account

1. Go to <https://pypi.org/account/register/>
2. Enable 2FA (required for new projects)
3. Create an API token at <https://pypi.org/manage/account/token/>

### 5. Upload to PyPI

```bash
twine upload dist/*
# Username: __token__
# Password: pypi-AgEI...   (your API token)
```

Or store credentials so you don't type them every time:

```bash
# ~/.pypirc
[pypi]
  username = __token__
  password = pypi-AgEI...
```

### 6. Test on TestPyPI first (recommended)

```bash
# Upload to test index
twine upload --repository testpypi dist/*

# Install from test index
pip install --index-url https://test.pypi.org/simple/ wizardai
```

### 7. Install from PyPI

Once published, anyone can install it with:

```bash
pip install wizardai
pip install "wizardai[all]"
```

### Releasing new versions

1. Bump `version` in `pyproject.toml`
2. `python -m build`
3. `twine upload dist/*`

> **Tip:** Use [bump-my-version](https://github.com/callowayproject/bump-my-version) or [commitizen](https://commitizen-tools.github.io/commitizen/) to automate version bumping.

---

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest`
5. Format code: `black . && isort .`
6. Open a pull request.

---

## License

MIT © WizardAI Contributors — see [LICENSE](LICENSE) for full text.
