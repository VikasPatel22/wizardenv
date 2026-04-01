# WizardAI SDK — Documentation

> **Version:** v1.0.0 · **License:** MIT · **API:** Sagittarius Labs

A powerful, all-in-one Python SDK for AI integration using the Sagittarius Labs API. Combines conversational AI, computer vision, speech I/O, memory management, and a flexible plugin system into a single importable file.

**Features:** Conversational AI · Computer Vision · Speech I/O · Memory Management · Plugin System

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [WizardAI (Orchestrator)](#wizardai)
4. [AIClient](#aiclient)
5. [AIResponse](#airesponse)
6. [ConversationAgent](#conversationagent)
7. [Pattern](#pattern)
8. [MemoryManager](#memorymanager)
9. [Message](#message)
10. [PluginBase](#pluginbase)
11. [PluginManager](#pluginmanager)
12. [Plugin Examples](#plugin-examples)
13. [VisionModule](#visionmodule)
14. [SpeechModule](#speechmodule)
15. [Logger](#logger)
16. [FileHelper](#filehelper)
17. [DataSerializer](#dataserializer)
18. [RateLimiter](#ratelimiter)
19. [Exceptions](#exceptions)
20. [Constants & Metadata](#constants--metadata)
21. [Full Examples](#full-examples)

---

## Installation

WizardAI is distributed as a single Python file hosted by Sagittarius Labs. Install it using one of the methods below — the file is placed directly into your Python `site-packages` directory so you can `import wizardai` from anywhere.

> **Install path:** `C:\Program Files\Python311\Lib\site-packages\wizardai.py`

### CMD (Admin)

```bat
:: Run Command Prompt as Administrator
:: Downloads wizardai.py from Sagittarius Labs and installs it

curl -L -o "%TEMP%\wizardai.py" ^
  "https://sagittarius1.netlify.app/infrastructure/downloads/scripts/site-packages/wizardai.py"

copy /Y "%TEMP%\wizardai.py" ^
  "C:\Program Files\Python311\Lib\site-packages\wizardai.py"

:: Verify installation
python -c "import wizardai; print(wizardai.__version__)"
```

### PowerShell (Admin)

```powershell
# Run PowerShell as Administrator

$url  = "https://sagittarius1.netlify.app/infrastructure/downloads/scripts/site-packages/wizardai.py"
$dest = "C:\Program Files\Python311\Lib\site-packages\wizardai.py"

Invoke-WebRequest -Uri $url -OutFile $dest

# Verify
python -c "import wizardai; print('Installed:', wizardai.__version__)"
```

### PowerShell (Legacy WebClient)

```powershell
# PowerShell (Administrator) — legacy WebClient method

$wc   = New-Object System.Net.WebClient
$url  = "https://sagittarius1.netlify.app/infrastructure/downloads/scripts/site-packages/wizardai.py"
$dest = "C:\Program Files\Python311\Lib\site-packages\wizardai.py"

$wc.DownloadFile($url, $dest)

# Confirm
Get-Item $dest
python -c "import wizardai; print(wizardai.__version__)"
```

### Batch Script

```bat
@echo off
:: Save this file as install_wizardai.bat
:: Right-click → Run as Administrator

setlocal
set URL=https://sagittarius1.netlify.app/infrastructure/downloads/scripts/site-packages/wizardai.py
set DEST=C:\Program Files\Python311\Lib\site-packages\wizardai.py

echo Downloading WizardAI SDK...
curl -L -o "%TEMP%\wizardai.py" "%URL%"

if %errorlevel% neq 0 (
    echo Download failed. Check internet connection.
    pause
    exit /b 1
)

echo Installing to site-packages...
copy /Y "%TEMP%\wizardai.py" "%DEST%"

echo Verifying...
python -c "import wizardai; print('WizardAI', wizardai.__version__, 'installed OK')"

pause
```

### curl.exe (Direct)

```bat
:: Using curl.exe directly (Windows 10 1803+ has curl built-in)
:: Run CMD or PowerShell as Administrator

curl.exe --location --progress-bar ^
  --output "C:\Program Files\Python311\Lib\site-packages\wizardai.py" ^
  "https://sagittarius1.netlify.app/infrastructure/downloads/scripts/site-packages/wizardai.py"

:: Test the import
python.exe -c "import wizardai; print(wizardai.__version__)"
```

> ⚠️ **Administrator Required:** Writing to `C:\Program Files\` requires elevated privileges. Always run your terminal as Administrator when installing.

### Optional Dependencies

WizardAI has zero required third-party dependencies at import time. Extra features need:

| Package | Required For | Install Command |
|---|---|---|
| `requests` | AIClient HTTP calls (core feature) | `pip install requests` |
| `opencv-python` | VisionModule — camera & image processing | `pip install opencv-python` |
| `SpeechRecognition` | SpeechModule STT (listen) | `pip install SpeechRecognition` |
| `pyttsx3` | SpeechModule TTS (pyttsx3 backend) | `pip install pyttsx3` |
| `gtts` | SpeechModule TTS (gTTS backend) | `pip install gtts` |
| `pygame` | Audio playback for gTTS / ElevenLabs | `pip install pygame` |
| `openai-whisper` | SpeechModule Whisper STT backend | `pip install openai-whisper` |
| `numpy` | Whisper STT audio array processing | `pip install numpy` |

### API Key

Get your free API key at [sagittarius-labs.pages.dev](https://sagittarius-labs.pages.dev). You can provide it in three ways:

```python
# Option 1: Pass directly
wiz = wizardai.WizardAI(api_key="YOUR_API_KEY")

# Option 2: Environment variable (recommended)
# CMD: set WIZARDAI_API_KEY=your_key_here
# PS:  $env:WIZARDAI_API_KEY = "your_key_here"
wiz = wizardai.WizardAI()  # picks up env var automatically

# Option 3: Set at runtime
wiz.set_api_key("YOUR_API_KEY")
```

---

## Quick Start

```python
import wizardai

# 1. Create and start a session
wiz = wizardai.WizardAI(api_key="YOUR_API_KEY")
wiz.start()

# 2. Direct LLM call
print(wiz.ask("What is the speed of light?"))

# 3. Add a pattern rule (no API call)
wiz.agent.add_pattern("hello", "Hello from WizardAI!")
print(wiz.chat("hello"))

# 4. Streaming response
for chunk in wiz.ai.chat_stream([{"role":"user","content":"Write a haiku"}]):
    print(chunk, end="", flush=True)

# 5. Long-term memory
wiz.remember("user_name", "Alice")
print(wiz.recall("user_name"))  # → Alice

# 6. Stop session
wiz.stop()

# Or use as a context manager
with wizardai.WizardAI(api_key="YOUR_API_KEY") as wiz:
    print(wiz.ask("Hello!"))
```

---

## WizardAI

**Class** — All-in-one AI session orchestrator. Supports context-manager protocol (`with WizardAI(...) as wiz:`).

The top-level orchestrator that bundles every WizardAI component into a single object. All subsystems — AI client, conversation agent, memory, speech, vision, and plugins — are accessible as attributes.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | Sagittarius Labs API key. Falls back to `WIZARDAI_API_KEY` env var. |
| `model` | `str \| None` | `None` | LLM model override. Default: `sagittarius/deep-vl-r1-128b`. |
| `max_tokens` | `int` | `1024` | Default maximum tokens for LLM responses. |
| `temperature` | `float` | `0.7` | Default sampling temperature (0 = deterministic). |
| `enable_vision` | `bool` | `False` | Open webcam on `start()`. Requires `opencv-python`. |
| `camera_device` | `int` | `0` | OpenCV camera device index. |
| `camera_width` | `int` | `640` | Capture width in pixels. |
| `camera_height` | `int` | `480` | Capture height in pixels. |
| `enable_speech` | `bool` | `False` | Initialise STT/TTS on `start()`. |
| `stt_backend` | `str` | `'google'` | `'google'` \| `'sphinx'` \| `'whisper'` |
| `tts_backend` | `str` | `'pyttsx3'` | `'pyttsx3'` \| `'gtts'` \| `'elevenlabs'` |
| `language` | `str` | `'en-US'` | BCP-47 language code. |
| `agent_name` | `str` | `'WizardBot'` | Display name of the conversation agent. |
| `fallback_response` | `str` | — | Response text when no pattern matches and LLM is unavailable. |
| `max_history` | `int` | `50` | Sliding window size for conversation memory. |
| `memory_path` | `str \| None` | `None` | Path for persistent memory JSON file. |
| `system_prompt` | `str \| None` | `None` | Default system prompt prepended to every LLM call. |
| `log_level` | `str` | `'INFO'` | `'DEBUG'` \| `'INFO'` \| `'WARNING'` \| `'ERROR'` |
| `log_file` | `str \| None` | `None` | Optional path to write logs to disk. |
| `data_dir` | `str` | `'./wizardai_data'` | Working directory for data persistence. |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `ai` | `AIClient` | Direct access to the Sagittarius Labs AI client. |
| `agent` | `ConversationAgent` | Pattern-matched conversation engine. |
| `memory` | `MemoryManager` | Conversation history and long-term memory. |
| `plugins` | `PluginManager` | Plugin registry and dispatcher. |
| `vision` | `VisionModule \| None` | Camera module (None if not enabled). |
| `speech` | `SpeechModule \| None` | Speech module (None if not enabled). |
| `files` | `FileHelper` | File I/O utilities rooted at `data_dir`. |
| `serializer` | `DataSerializer` | JSON / pickle serialization helpers. |

### Methods

**`start()` → `None`**
Open camera, initialise speech, call `on_start()` on all plugins. Must be called before `chat()` or `ask()` if using vision/speech.

**`stop()` → `None`**
Release camera, stop listening threads, save memory, and call `on_stop()` on all plugins.

**`chat(user_input: str)` → `str`**
Full pipeline: plugins → pattern matching → LLM fallback. Returns response string. All messages are logged to memory.

**`ask(prompt, model=None, max_tokens=None, temperature=None, system_prompt=None, include_history=True, image_b64=None)` → `str`**
Bypass pattern matching and send directly to the LLM. Optionally pass a base64 image for multimodal queries. Returns generated text string.

**`ask_raw(prompt: str, **kwargs)` → `AIResponse`**
Like `ask()` but returns the full `AIResponse` dataclass including token usage and latency.

**`listen(timeout=5.0)` → `str | None`**
Capture and transcribe speech from the microphone. Returns transcribed text or `None` on failure. Requires `enable_speech=True`.

**`say(text: str, blocking=True)` → `None`**
Speak text aloud using the configured TTS engine. Requires `enable_speech=True`.

**`voice_chat(timeout=5.0)` → `str | None`**
Listen → process through `chat()` → speak response back. Full voice loop in one call.

**`capture()` → `ndarray | None`**
Capture and return a single BGR frame from the camera. Requires `enable_vision=True`.

**`snapshot(path='snapshot.jpg')` → `Path | None`**
Capture a frame and save it to disk. Returns the `Path` of the saved file.

**`remember(key: str, value: Any)` → `None`**
Store a fact in long-term persistent memory.

**`recall(key: str, default=None)` → `Any`**
Retrieve a fact from long-term memory. Returns `default` if not found.

**`get_history(n=10)` → `List[Dict]`**
Return the last `n` conversation turns as a list of dicts.

**`add_plugin(plugin_cls, config=None)` → `PluginBase`**
Register a plugin class with the plugin manager.

**`load_plugins_from_dir(directory)` → `List[PluginBase]`**
Auto-discover and load all plugin Python files from a directory.

**`set_system_prompt(prompt: str)` → `None`**
Update the default system prompt for all future LLM calls.

**`set_model(model: str)` → `None`**
Switch the active model at runtime.

**`set_api_key(api_key: str)` → `None`**
Update the API key at runtime without recreating the client.

**`run_repl(prompt_str='You: ', quit_commands=None, voice_mode=False)` → `None`**
Start an interactive REPL in the terminal. Handles Ctrl+C gracefully. Set `voice_mode=True` to use voice I/O instead of keyboard.

---

## AIClient

**Class** — Low-level HTTP client for the Sagittarius Labs API. Handles authentication, rate limiting, streaming, and automatic retries.

> 📡 **Endpoint:** All requests go to `https://sagittarius-labs.pages.dev/api/chat` using model `sagittarius/deep-vl-r1-128b`.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str \| None` | `None` | API key. Falls back to `WIZARDAI_API_KEY` env var. |
| `model` | `str` | `sagittarius/deep-vl-r1-128b` | Default model identifier. |
| `max_retries` | `int` | `3` | Retry attempts on transient errors. |
| `retry_delay` | `float` | `1.0` | Initial retry delay in seconds (doubles each attempt). |
| `timeout` | `float` | `60.0` | HTTP request timeout in seconds. |
| `rate_limit_calls` | `int` | `60` | Max API calls per window. |
| `rate_limit_period` | `float` | `60.0` | Rate-limit window in seconds. |
| `logger` | `Logger \| None` | `None` | Optional Logger instance. |

### Methods

**`chat(messages, model=None, max_tokens=1024, temperature=0.7, system_prompt=None, **kwargs)` → `AIResponse`**
Multi-turn chat (non-streaming). `messages` is a list of `{"role": ..., "content": ...}` dicts. Retries on transient errors automatically.

**`chat_stream(messages, model=None, max_tokens=1024, temperature=0.7, system_prompt=None)` → `Generator[str]`**
Multi-turn streaming — yields text chunks as they arrive from the API. Use `for chunk in client.chat_stream(...): print(chunk, end="")`.

**`complete(prompt: str, **kwargs)` → `AIResponse`**
Single-turn convenience wrapper — pass a plain string, get an `AIResponse` back.

**`complete_stream(prompt: str, **kwargs)` → `Generator[str]`**
Single-turn streaming convenience wrapper.

**`set_api_key(api_key: str)` → `None`**
Update the API key at runtime.

**`set_model(model: str)` → `None`**
Change the default model.

### Example — Streaming

```python
from wizardai import AIClient

client = AIClient(api_key="YOUR_KEY")

# Non-streaming
resp = client.complete("Explain quantum entanglement simply.")
print(resp.text)
print(f"Tokens used: {resp.usage}")
print(f"Latency: {resp.latency_ms:.0f}ms")

# Streaming
for chunk in client.complete_stream("Tell me a story"):
    print(chunk, end="", flush=True)

# Multi-turn chat
messages = [
    {"role": "user",      "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
    {"role": "user",      "content": "What's my name?"},
]
resp = client.chat(messages, system_prompt="You are a friendly assistant.")
print(resp.text)  # → Your name is Alice.
```

---

## AIResponse

**Dataclass** — Structured response returned by `AIClient.chat()` and `AIClient.complete()`. Supports `str(response)` which returns `response.text`.

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Generated text content from the model. |
| `model` | `str` | Model identifier that produced the response. |
| `usage` | `Dict[str, int]` | Token usage stats: `prompt_tokens`, `completion_tokens`, `total_tokens`. |
| `raw` | `Dict[str, Any]` | Full raw JSON response from the API. |
| `latency_ms` | `float` | Round-trip HTTP latency in milliseconds. |

---

## ConversationAgent

**Class** — AIML-style rule-based chat engine. Matches user input against registered patterns using wildcards and named capture groups. Falls back to a default message when no pattern matches.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'WizardBot'` | Agent display name. |
| `fallback` | `str` | — | Response when no pattern matches. |
| `memory` | `MemoryManager \| None` | `None` | Shared memory manager. |
| `logger` | `Logger \| None` | `None` | Optional logger. |
| `case_sensitive` | `bool` | `False` | Whether pattern matching is case-sensitive. |

### Pattern Wildcards

| Wildcard | Matches | Template Ref |
|---|---|---|
| `*` | One or more words (greedy) | `{wildcard}` or `{0}` |
| `?` | Exactly one word | `{0}` |
| `{name}` | Named capture group | `{name}` |

### Methods

**`add_pattern(pattern, template, priority=0, context=None, tags=None)` → `Pattern`**
Register a pattern rule. `template` can be a string, callable, or list of alternatives (random pick). Higher `priority` matches first.

**`respond(user_input: str)` → `str`**
Process input through all registered patterns and return a response. Runs pre/post processors. Logs to memory.

**`load_patterns_from_dict(rules: Dict)` → `None`**
Bulk-load patterns from a `{"pattern": "template"}` dict.

**`load_patterns_from_file(path: str)` → `None`**
Load patterns from a JSON file on disk.

**`add_preprocessor(fn: Callable[[str], str])` → `None`**
Register a function to transform input text before pattern matching.

**`add_postprocessor(fn: Callable[[str], str])` → `None`**
Register a function to transform the response text after matching.

**`register_plugin(name: str, handler: Callable)` → `None`**
Register an inline `!name`-invocable plugin. Users type `!name args` to trigger it.

**`set_context(context: str)` → `None`**
Set the active context. Only patterns with matching `context` (or none) will fire.

**`search_history(query: str, top_k=5)` → `List[Tuple[Message, float]]`**
Simple keyword-overlap search over conversation history. Returns `(message, score)` tuples sorted by relevance.

**`reset()` → `None`**
Clear conversation history and reset the active context.

### Example — Patterns

```python
from wizardai import ConversationAgent

agent = ConversationAgent(name="MyBot")

# Static response
agent.add_pattern("hello", "Hello! How can I help?")

# Wildcard — {wildcard} contains captured text
agent.add_pattern("my name is *", "Nice to meet you, {wildcard}!")

# Named capture group
agent.add_pattern("call me {name}", "Sure, I'll call you {name}.")

# Random alternatives
agent.add_pattern("how are you", ["I'm great!", "Ready to help!", "All good!"])

# Callable template
import datetime
agent.add_pattern("what time is it", lambda: f"It is {datetime.datetime.now():%H:%M}.")

# Priority — matched before lower-priority rules
agent.add_pattern("hello *", "Hey {wildcard}!", priority=10)

# Inline plugin (!calc expression)
agent.register_plugin("calc", lambda args: eval(args))

print(agent.respond("hello"))
print(agent.respond("!calc 3 * 7"))  # → 21
```

---

## Pattern

**Dataclass** — A single conversation rule registered with `ConversationAgent`.

| Field | Type | Description |
|---|---|---|
| `pattern` | `str` | Input pattern string. Supports `*`, `?`, and `{name}` wildcards. |
| `template` | `str \| Callable \| List[str]` | Response string, callable returning a string, or list of alternatives. |
| `priority` | `int` | Higher values are matched first. Default: 0. |
| `context` | `str \| None` | Context key required for this rule to fire. `None` fires in any context. |
| `tags` | `List[str]` | Arbitrary labels for filtering and introspection. |

---

## MemoryManager

**Class** — Manages both short-term conversation history (sliding window deque) and long-term key-value memory (persisted to JSON). Also provides an ephemeral session context dict.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_history` | `int` | `50` | Maximum messages kept in sliding window. |
| `persist_path` | `str \| Path \| None` | `None` | JSON file path for persistence. Auto-saves on every change. |
| `logger` | `Logger \| None` | `None` | Optional logger. |

### Short-term History

**`add_message(role, content, metadata=None)` → `Message`**
Add a message to the sliding window. `role`: `'user'` | `'assistant'` | `'system'`.

**`get_history(n=None, role_filter=None)` → `List[Message]`**
Return messages. Optionally limit to last `n` or filter by role.

**`get_messages_for_api(n=None, include_system=True)` → `List[Dict[str, str]]`**
Return history formatted as `[{"role":…, "content":…}]` dicts, ready to pass to the API.

**`search_history(query: str, top_k=5)` → `List[Tuple[Message, float]]`**
Keyword-overlap search over history. Returns `(message, relevance_score)`.

**`last_message(role=None)` → `Message | None`**
Return the most recent message, optionally filtered by role.

**`clear_history()` → `None`**
Clear all conversation history from the sliding window.

### Long-term Memory

**`remember(key: str, value: Any)` → `None`**
Store an arbitrary value. Persisted to JSON immediately if `persist_path` is set.

**`recall(key: str, default=None)` → `Any`**
Retrieve a stored value by key.

**`forget(key: str)` → `bool`**
Delete a key from long-term memory. Returns `True` if found and deleted.

**`list_memories()` → `List[str]`**
Return all stored long-term memory keys.

### Ephemeral Context

**`set_context(key, value)` / `get_context(key, default=None)` / `clear_context()`**
In-session only key-value store. Not persisted to disk.

### Persistence

**`save(path=None)` / `load(path=None)`**
Manually save or load from a JSON file. Auto-save occurs on every change when `persist_path` is configured.

---

## Message

**Class** — A single conversation message stored in `MemoryManager`. Supports serialization via `to_dict()` and `Message.from_dict()`.

| Attribute | Type | Description |
|---|---|---|
| `role` | `str` | `'user'` \| `'assistant'` \| `'system'` |
| `content` | `str` | Message text content. |
| `timestamp` | `float` | Unix timestamp when the message was created. |
| `metadata` | `Dict[str, Any]` | Arbitrary key-value pairs attached to the message. |

---

## PluginBase

**Abstract Class** — Base class for all WizardAI plugins. Subclass it, implement `on_message()`, set the class attributes, and register with `PluginManager`.

### Class Attributes (override in subclass)

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Unique plugin identifier. Used as the registry key. |
| `description` | `str` | Human-readable description. |
| `version` | `str` | Semantic version string. |
| `author` | `str` | Author name. |
| `triggers` | `List[str]` | Informational list of trigger phrases (documentation only). |

### Instance Attributes

| Attribute | Type | Description |
|---|---|---|
| `config` | `Dict[str, Any]` | Plugin configuration dict passed at registration. |
| `logger` | `Logger` | Logger instance for the plugin. |
| `is_enabled` | `bool` (property) | Whether the plugin is active. |

### Methods to Override

**`on_message(text: str, context: Dict)` → `str | None`** *(Abstract — must implement)*
Process user text and return a response string, or `None` to pass through to the next plugin/agent.

**`setup()`**
Called once after `__init__`. Override to initialise resources (database connections, models, etc.).

**`teardown()`**
Called when the plugin is unregistered. Override to release resources.

**`on_start()` / `on_stop()`**
Called when the WizardAI session starts/stops.

**`enable()` / `disable()`**
Enable or disable the plugin. Disabled plugins are skipped by `PluginManager.dispatch()`.

---

## PluginManager

**Class** — Manages plugin lifecycle: registration, dispatch, file/directory loading, and bulk start/stop.

**`register(plugin_cls, config=None, name_override=None)` → `PluginBase`**
Instantiate and register a `PluginBase` subclass. Raises `PluginError` if already registered.

**`unregister(name: str)` → `bool`**
Remove a plugin and call its `teardown()`.

**`dispatch(text, context=None)` → `str | None`**
Pass text to each enabled plugin in registration order. Returns the first non-`None` response.

**`dispatch_all(text, context=None)` → `List[Tuple[str, str]]`**
Call all enabled plugins and collect every non-`None` response as `(plugin_name, response)`.

**`load_from_file(path, config=None)` → `PluginBase`**
Dynamically import a `.py` file and register the first `PluginBase` subclass found in it.

**`load_from_directory(directory, config=None)` → `List[PluginBase]`**
Auto-discover and load all `.py` files in a directory (skipping `_`-prefixed files).

**`get(name: str)` → `PluginBase | None`**
Retrieve a plugin instance by name.

**`list_plugins(enabled_only=False)` → `List[PluginBase]`**
Return all (or only enabled) registered plugins.

**`start_all()` / `stop_all()`**
Call `on_start()` / `on_stop()` on all enabled plugins. Called automatically by `WizardAI.start()` and `stop()`.

---

## Plugin Examples

```python
from wizardai import PluginBase, PluginManager
import random

class JokePlugin(PluginBase):
    name        = "jokes"
    description = "Tells jokes on demand."
    version     = "1.0.0"
    author      = "WizardAI Dev"
    triggers    = ["joke", "tell me a joke", "make me laugh"]

    def setup(self):
        self.jokes = [
            "Why do Python devs wear glasses? They can't C!",
            "What's a computer's favourite snack? Microchips.",
            "Why was the web developer stressed? Too many tabs.",
        ]

    def on_message(self, text, context):
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.triggers):
            return random.choice(self.jokes)
        return None  # pass to next plugin

# Register and use
manager = PluginManager()
manager.register(JokePlugin)

print(manager.dispatch("tell me a joke"))
print(manager.dispatch("hello there"))  # → None (no match)
```

---

## VisionModule

**Class** — Real-time camera access and image processing via OpenCV. Requires `pip install opencv-python`. Supports context manager protocol.

> ⚠️ **Optional Dependency:** Requires `opencv-python`. Attempting to use without it raises `VisionError` with install instructions.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `device_id` | `int` | `0` | OpenCV camera device index. |
| `width` | `int` | `640` | Capture width in pixels. |
| `height` | `int` | `480` | Capture height in pixels. |
| `fps` | `int` | `30` | Target frames per second for streaming. |
| `logger` | `Logger \| None` | `None` | Optional logger. |

### Capture Methods

**`open()` / `close()` / `is_open()`**
Open or release the camera device.

**`capture_frame()` → `ndarray`**
Capture a single BGR frame. Raises `VisionError` if camera is not open.

**`capture_frames(n, delay=0.0)` → `List[ndarray]`**
Capture `n` frames with an optional delay between each.

**`save_frame(frame, path, quality=95)` → `Path`**
Save a frame to disk. JPEG quality 0–100.

**`load_image(path)` → `ndarray`**
Load an image from disk as a BGR ndarray.

### Image Processing

**`resize_frame(frame, width, height)` / `to_grayscale(frame)` / `to_rgb(frame)` / `flip(frame, axis=1)`**
Standard image transformations. Return a new ndarray.

**`draw_rectangle(frame, x, y, w, h, colour=(0,255,0), thickness=2)`**
Draw a coloured rectangle on the frame in-place.

**`draw_text(frame, text, x, y, font_scale=0.7, colour=(0,255,0), thickness=2)`**
Overlay text on a frame using OpenCV's Hershey Simplex font.

**`encode_to_base64(frame, ext='.jpg')` → `str`**
Encode a frame as a base64 string. Useful for passing images to the LLM via `ask(image_b64=...)`.

### Face Detection

**`detect_faces(frame, scale_factor=1.1, min_neighbours=5, min_size=(30,30))` → `List[Dict]`**
Haar cascade face detection. Returns list of `{"x", "y", "w", "h"}` dicts.

**`annotate_faces(frame)` → `Tuple[ndarray, List[Dict]]`**
Detect faces and draw bounding boxes + labels. Returns the annotated frame and face list.

### Streaming

**`start_stream(callback=None, show_preview=False)`**
Start a background thread that captures frames and calls registered callbacks. Pass `show_preview=True` to open an OpenCV window; press `q` to quit.

**`add_frame_callback(callback: Callable[[ndarray], None])`**
Register a function called with each captured frame during streaming.

**`stop_stream()`**
Stop the background streaming thread.

### Example — Vision

```python
from wizardai import VisionModule

# Context-manager usage
with VisionModule(device_id=0, width=1280, height=720) as cam:
    frame = cam.capture_frame()
    cam.save_frame(frame, "snapshot.jpg", quality=90)

    # Face detection
    annotated, faces = cam.annotate_faces(frame)
    print(f"Found {len(faces)} face(s)")
    cam.save_frame(annotated, "faces.jpg")

    # Send frame to AI for vision query
    b64 = cam.encode_to_base64(frame)
    # wiz.ask("Describe this image", image_b64=b64)

    # Live stream with callback
    def on_frame(f):
        _, face_list = cam.annotate_faces(f)
        if face_list:
            print("Face detected!")

    cam.start_stream(callback=on_frame, show_preview=True)
```

---

## SpeechModule

**Class** — Speech-to-text (STT) and text-to-speech (TTS) with multiple backends. Supports continuous background listening via a daemon thread.

**STT Backends:** `google` (Google Speech API, online) · `sphinx` (CMU Sphinx, offline) · `whisper` (OpenAI Whisper, offline/GPU)

**TTS Backends:** `pyttsx3` (local system voice, offline) · `gtts` (Google TTS, online MP3) · `elevenlabs` (ElevenLabs API, premium voices)

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stt_backend` | `str` | `'google'` | STT backend: `'google'` \| `'sphinx'` \| `'whisper'` |
| `tts_backend` | `str` | `'pyttsx3'` | TTS backend: `'pyttsx3'` \| `'gtts'` \| `'elevenlabs'` |
| `language` | `str` | `'en-US'` | BCP-47 language code. |
| `tts_rate` | `int` | `150` | Speech rate (words per minute) for pyttsx3. |
| `tts_volume` | `float` | `1.0` | Volume (0.0–1.0) for pyttsx3. |
| `elevenlabs_api_key` | `str \| None` | `None` | ElevenLabs API key (falls back to `ELEVENLABS_API_KEY` env var). |
| `elevenlabs_voice_id` | `str \| None` | `'21m00…'` | ElevenLabs voice ID. |

### STT Methods

**`listen(timeout=5.0, phrase_time_limit=15.0, adjust_noise=True, device_index=None)` → `str`**
Capture audio from microphone and return transcribed text. Raises `SpeechError` on failure.

**`transcribe_file(path)` → `str`**
Transcribe a pre-recorded audio file.

**`list_microphones()` → `List[Dict]`**
Return all available microphone devices as `{"index", "name"}` dicts.

### TTS Methods

**`say(text: str, blocking=True)` → `str | None`**
Speak text aloud. For file-based backends (gTTS, ElevenLabs) returns the temp file path.

**`synthesise_to_file(text, path)` → `Path`**
Synthesise speech and save to an audio file without playing it.

**`set_tts_rate(rate: int)` / `set_tts_volume(volume: float)` / `set_tts_voice(voice_id: str)`**
Adjust pyttsx3 TTS properties at runtime.

**`list_voices()` → `List[Dict]`**
Return available pyttsx3 voice IDs and names.

### Continuous Listening

**`start_continuous_listening(callback=None, timeout=None, phrase_time_limit=10.0)`**
Start a background daemon thread that continuously listens and calls registered callbacks with transcribed text.

**`add_listener(callback: Callable[[str], None])`**
Register a callback invoked with every transcribed utterance during continuous listening.

**`stop_continuous_listening()`**
Stop the background listening thread.

### Example — Voice Assistant Loop

```python
from wizardai import WizardAI

wiz = WizardAI(
    api_key="YOUR_KEY",
    enable_speech=True,
    stt_backend="google",
    tts_backend="pyttsx3",
    language="en-US",
)
wiz.start()

# Single voice exchange
response = wiz.voice_chat(timeout=8.0)
print(f"Bot said: {response}")

# Continuous background listener
def handle_speech(text):
    reply = wiz.chat(text)
    wiz.say(reply)

wiz.speech.start_continuous_listening(callback=handle_speech)
# ... your app logic ...
wiz.speech.stop_continuous_listening()
wiz.stop()
```

---

## Logger

**Class** — Configurable coloured terminal logger backed by Python's `logging` module. Optionally writes to a log file.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | `'wizardai'` | Logger name (appears in output). |
| `level` | `str` | `'INFO'` | `'DEBUG'` \| `'INFO'` \| `'WARNING'` \| `'ERROR'` \| `'CRITICAL'` |
| `log_file` | `str \| None` | `None` | Optional file path to mirror logs. |
| `coloured` | `bool` | `True` | ANSI colour coding in terminal. |

**Methods:** `debug(msg)` · `info(msg)` · `warning(msg)` · `error(msg)` · `critical(msg)` · `set_level(level)`

```python
log = wizardai.Logger("my_app", level="DEBUG", log_file="app.log")
log.info("Session started")
log.warning("Memory usage high")
log.error("Connection failed")
log.set_level("WARNING")
```

---

## FileHelper

**Class** — High-level file I/O rooted at a `base_dir`. All paths are resolved relative to `base_dir` unless absolute.

| Method | Signature | Description |
|---|---|---|
| `write_text` | `(path, content, encoding='utf-8', append=False) → Path` | Write a string to a text file. |
| `read_text` | `(path, encoding='utf-8') → str` | Read a text file to a string. |
| `read_lines` | `(path, strip=True) → List[str]` | Read all lines, optionally stripping whitespace. |
| `write_json` | `(path, data, indent=2) → Path` | Serialize data to a JSON file. |
| `read_json` | `(path) → Any` | Parse a JSON file and return the object. |
| `write_csv` | `(path, rows, fieldnames=None) → Path` | Write a list of dicts to a CSV file. |
| `read_csv` | `(path) → List[Dict]` | Read a CSV file as a list of dicts. |
| `copy` | `(src, dst) → Path` | Copy a file. |
| `delete` | `(path) → bool` | Delete a file. Returns `True` if it existed. |
| `list_files` | `(directory='.', pattern='*', recursive=False) → List[Path]` | List files matching a glob pattern. |
| `timestamp_filename` | `(name, ext='') → str` | Generate a timestamped filename like `name_20250101_120000.ext`. |
| `resolve` | `(path) → Path` | Resolve a path relative to `base_dir`. |
| `ensure_dir` | `(path) → Path` | Create a directory (including parents) if it doesn't exist. |

---

## DataSerializer

**Class** — Serialize/deserialize Python objects to JSON, Pickle, or gzip-compressed variants. Format auto-detected from file extension.

| Method | Signature | Description |
|---|---|---|
| `save` | `(data, path, compress=False, indent=2) → Path` | Save data. Format from extension: `.json` / `.json.gz` / `.pkl` / `.pkl.gz` |
| `load` | `(path) → Any` | Load data. Format auto-detected from extension. |
| `to_json_string` | `(data, indent=2) → str` | Serialize to a JSON string. |
| `from_json_string` | `(text) → Any` | Deserialize from a JSON string. |
| `iter_jsonl` | `(path) → Iterator[Any]` | Iterate over records in a JSON-Lines file. |
| `write_jsonl` | `(path, records) → Path` | Write a list of objects as a JSON-Lines file. |

---

## RateLimiter

**Class** — Token-bucket rate limiter. Supports context-manager protocol. Used internally by `AIClient`.

### `__init__` Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_calls` | `int` | `60` | Maximum calls allowed per window. |
| `period` | `float` | `60.0` | Sliding window duration in seconds. |

**`wait()`**
Block until a slot is available in the current window. Call before every rate-limited operation.

**`is_allowed()` → `bool`**
Return `True` if a call can be made immediately without waiting.

```python
limiter = wizardai.RateLimiter(max_calls=10, period=60)

for item in large_list:
    limiter.wait()  # blocks if needed
    process(item)

# Context-manager style
with limiter:
    api_call()
```

---

## Exceptions

All WizardAI exceptions inherit from `WizardAIError` which has a `message` attribute and an optional HTTP `code`.

| Exception | Parent | Description |
|---|---|---|
| `WizardAIError` | `Exception` | Base exception for all WizardAI errors. Has `.message` and `.code` attributes. |
| `APIError` | `WizardAIError` | Raised when an AI API call fails. Includes HTTP status code. |
| `AuthenticationError` | `APIError` | API key is missing or invalid. Includes URL to obtain/verify key. Never retried. |
| `RateLimitError` | `APIError` | Rate limit exceeded (HTTP 429). Has `.retry_after` float in seconds. |
| `VisionError` | `WizardAIError` | Camera or image processing failure. |
| `CameraNotFoundError` | `VisionError` | Camera device index not found. Has `.device_id` attribute. |
| `SpeechError` | `WizardAIError` | Speech recognition or TTS failure. |
| `MicrophoneNotFoundError` | `SpeechError` | No microphone device detected. |
| `ConversationError` | `WizardAIError` | Conversation engine internal error. |
| `PluginError` | `WizardAIError` | Plugin load or execution failure. Has `.plugin_name` attribute. |
| `ConfigurationError` | `WizardAIError` | SDK misconfiguration detected. |

### Error Handling Pattern

```python
from wizardai import WizardAI, AuthenticationError, RateLimitError, APIError

try:
    response = wiz.ask("Hello")
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
    print("→ Visit https://sagittarius-labs.pages.dev/ for a key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except APIError as e:
    print(f"API error {e.code}: {e.message}")
except Exception as e:
    print(f"Unexpected: {e}")
```

---

## Constants & Metadata

| Name | Value | Description |
|---|---|---|
| `__version__` | `'1.0.0'` | SDK version string. |
| `__author__` | `'WizardAI Contributors'` | Author(s). |
| `__license__` | `'MIT'` | License identifier. |
| `_BASE_URL` | `'https://sagittarius-labs.pages.dev'` | Sagittarius Labs base URL. |
| `_ENDPOINT` | `'https://sagittarius-labs.pages.dev/api/chat'` | API chat endpoint. |
| `_MODEL` | `'sagittarius/deep-vl-r1-128b'` | Default model identifier. |
| `_SIGNUP_URL` | `'https://sagittarius-labs.pages.dev'` | URL to obtain an API key. |
| `_ENV_KEY` | `'WIZARDAI_API_KEY'` | Environment variable name for the API key. |

---

## Full Examples

### 1. Interactive Terminal REPL

```python
import wizardai

wiz = wizardai.WizardAI(
    api_key="YOUR_KEY",
    agent_name="Sage",
    system_prompt="You are Sage, a wise and helpful AI assistant.",
    memory_path="./session_memory.json",
    log_level="WARNING",
)
wiz.start()
wiz.run_repl(prompt_str="You: ", quit_commands=["quit", "/q", "exit"])
```

### 2. Multimodal Vision + AI

```python
import wizardai

with wizardai.WizardAI(api_key="YOUR_KEY", enable_vision=True) as wiz:
    frame = wiz.capture()
    b64   = wiz.vision.encode_to_base64(frame)
    answer = wiz.ask("Describe what you see in this image.", image_b64=b64)
    print(answer)
    wiz.snapshot("./captured.jpg")
```

### 3. Memory-Aware Chatbot

```python
import wizardai

wiz = wizardai.WizardAI(api_key="YOUR_KEY", memory_path="./memory.json")
wiz.start()

# Store facts
wiz.remember("user_name", "Alice")
wiz.remember("user_city", "London")
wiz.remember("preferences", {"lang": "en", "theme": "dark"})

# Recall later
name = wiz.recall("user_name")  # → 'Alice'
wiz.agent.add_pattern(
    "what is my name",
    f"Your name is {name}.",
)
print(wiz.chat("what is my name"))  # → Your name is Alice.

# Search history
results = wiz.memory.search_history("name", top_k=3)
for msg, score in results:
    print(f"[{score:.2f}] {msg.role}: {msg.content[:50]}")

wiz.stop()
```

### 4. Plugin System — Complete Example

```python
import wizardai
import datetime

class TimePlugin(wizardai.PluginBase):
    name        = "time"
    description = "Returns current time and date."
    version     = "1.0.0"
    triggers    = ["time", "date", "what time"]

    def on_message(self, text, context):
        text_l = text.lower()
        if "time" in text_l or "date" in text_l:
            now = datetime.datetime.now()
            return f"It is {now:%A, %B %d %Y} at {now:%H:%M:%S}."
        return None

class WeatherPlugin(wizardai.PluginBase):
    name        = "weather"
    description = "Stub weather plugin."
    version     = "1.0.0"

    def on_message(self, text, context):
        if "weather" in text.lower():
            return "It's sunny and 22°C. Perfect for coding!"
        return None

with wizardai.WizardAI(api_key="YOUR_KEY") as wiz:
    wiz.add_plugin(TimePlugin)
    wiz.add_plugin(WeatherPlugin)

    print(wiz.chat("what time is it"))     # → TimePlugin
    print(wiz.chat("what's the weather"))  # → WeatherPlugin
    print(wiz.chat("explain relativity"))  # → LLM fallback
```

### 5. Data Persistence & Serialization

```python
from wizardai import FileHelper, DataSerializer

fh = FileHelper(base_dir="./data")
ds = DataSerializer()

# Write and read text
fh.write_text("notes.txt", "WizardAI is awesome!")
print(fh.read_text("notes.txt"))

# Write and read JSON
fh.write_json("config.json", {"model": "deep-vl", "temp": 0.7})
cfg = fh.read_json("config.json")

# CSV round-trip
rows = [{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]
fh.write_csv("scores.csv", rows)
print(fh.read_csv("scores.csv"))

# Compressed pickle
ds.save({"big": "data"}, "./cache.pkl.gz")
data = ds.load("./cache.pkl.gz")

# JSON-Lines
ds.write_jsonl("./records.jsonl", [{"id": 1}, {"id": 2}])
for rec in ds.iter_jsonl("./records.jsonl"):
    print(rec)
```

### 6. Set WIZARDAI_API_KEY in Windows

**CMD (persistent):**
```bat
:: Set permanently for current user (persists after reboot)
setx WIZARDAI_API_KEY "your_api_key_here"

:: Restart CMD after running setx, then verify
echo %WIZARDAI_API_KEY%
```

**PowerShell (persistent):**
```powershell
# Set permanently for current user
[System.Environment]::SetEnvironmentVariable(
    "WIZARDAI_API_KEY",
    "your_api_key_here",
    "User"
)

# Verify in a new terminal
$env:WIZARDAI_API_KEY
```

**Temporary (session only):**
```bat
:: CMD — session only (gone when window closes)
set WIZARDAI_API_KEY=your_api_key_here
```
```powershell
# PowerShell — session only
$env:WIZARDAI_API_KEY = "your_api_key_here"
```

---

*WizardAI SDK v1.0.0 · MIT License · [sagittarius-labs.pages.dev](https://sagittarius-labs.pages.dev) · Model: `sagittarius/deep-vl-r1-128b`*
