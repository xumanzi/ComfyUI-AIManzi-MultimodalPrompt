"""AI蛮子多模态提示词节点。

界面只暴露模型、NInfer 开关、文字和媒体。推理服务路径等一次性配置放在
config/settings.json，避免把硬件/服务参数塞进工作流。
"""
from __future__ import annotations

import base64
import gc
import http.client
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from urllib.parse import urlsplit
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    import torch
except ImportError:  # ComfyUI always provides torch; keeps module error readable.
    torch = None


PLUGIN_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = PLUGIN_DIR / "config" / "settings.json"
MODEL_EXTENSIONS = {".ninfer", ".gguf"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
AUTO_CONTEXT_MIN = 4096
AUTO_CONTEXT_MAX = 262144
MAX_DYNAMIC_IMAGE_INPUTS = 10
SKILL_MAX_FILES = 500
SKILL_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
SKILL_MAX_TEXT_FILE_BYTES = 1024 * 1024
SKILL_MAX_TEXT_BYTES = 8 * 1024 * 1024
SKILL_TEXT_EXTENSIONS = {".md", ".markdown", ".txt", ".json", ".yaml", ".yml"}
# GGUF can use the broader shared budget. NInfer is more reliable when each
# image stays near its usual ~1024 visual-token geometry, so it also receives a
# per-item cap rather than allowing one image to consume the whole envelope.
VISION_SAFE_TOTAL_PIXELS = 6 * 1024 * 1024
NINFER_VISION_LEVELS = (
    (4 * 1024 * 1024, 1 * 1024 * 1024),
    (2 * 1024 * 1024, 512 * 1024),
    (1 * 1024 * 1024, 256 * 1024),
)


class _DynamicImageOptionalInputs(dict):
    """Declare the image sockets that the browser adds after node creation.

    ComfyUI validates graph links against ``INPUT_TYPES`` before it calls
    ``generate``.  Plain dictionaries therefore silently discard 图像_2 and
    later sockets even though the canvas displays them.  This mapping retains
    the normal optional sockets while explicitly accepting 图像_1 through
    图像_10 as IMAGE inputs during that validation step.
    """

    def __contains__(self, key: object) -> bool:
        return super().__contains__(key) or (
            isinstance(key, str)
            and (match := re.fullmatch(r"图像_(\d+)", key)) is not None
            and 1 <= int(match.group(1)) <= MAX_DYNAMIC_IMAGE_INPUTS
        )

    def __getitem__(self, key: str):
        if super().__contains__(key):
            return super().__getitem__(key)
        if key in self:
            return ("IMAGE",)
        raise KeyError(key)


def _settings() -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "llm_roots": [str(Path.cwd() / "models" / "LLM")],
        "ninfer_api_base": "http://127.0.0.1:8080/v1",
        "llama_api_base": "http://127.0.0.1:8082/v1",
        "ffmpeg": "ffmpeg",
        "request_timeout_seconds": 600,
        "llama_server_command": "",
    }
    try:
        defaults.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        pass
    return defaults


def _auto_context_tokens(text: str, image_count: int = 0, required_tokens: int | None = None) -> int:
    """Choose the smallest power-of-two context that can carry this request and its output.

    Text is conservatively estimated before a model is loaded; exact token overflow retries pass
    required_tokens here. Media has a reserve because vision token counts are model-dependent.
    """
    if required_tokens is None:
        estimated_input = max(len(text), int(len(text.encode("utf-8")) * 0.80)) + image_count * 2048
    else:
        estimated_input = required_tokens
    needed = estimated_input + max(AUTO_CONTEXT_MIN, estimated_input) + 512
    context = AUTO_CONTEXT_MIN
    while context < needed and context < AUTO_CONTEXT_MAX:
        context *= 2
    return min(context, AUTO_CONTEXT_MAX)


def _roots() -> list[Path]:
    result: list[Path] = []
    for item in _settings()["llm_roots"]:
        path = Path(item)
        if path.exists():
            result.append(path)
    return result


def _scan_models() -> list[str]:
    models: list[str] = []
    for root in _roots():
        for path in root.rglob("*"):
            if (path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
                    and not path.name.lower().startswith("mmproj")):
                models.append(str(path))
    return sorted(models, key=lambda p: (Path(p).suffix.lower(), Path(p).name.lower())) or ["未找到模型"]


def _vision_candidates() -> list[str]:
    candidates: list[str] = ["自动（使用主模型，如具备视觉能力）"]
    for model in _scan_models():
        lower = model.lower()
        parent = Path(model).parent
        is_gguf_vision = Path(model).suffix.lower() == ".gguf" and bool(any(parent.glob("*mmproj*.gguf")))
        if is_gguf_vision or any(token in lower for token in ("vl", "vision", "florence", "llava", "cog")):
            candidates.append(model)
    return candidates


def _model_choices() -> list[str]:
    """The canvas shows only filenames; execution resolves them to an LLM-root path."""
    names = [_display_model_name(Path(item)) for item in _scan_models()]
    # Keep this legacy sentinel in the schema so old workflows with a value displaced from
    # the former mmproj widget pass ComfyUI's pre-execution combo validation.
    return ["自动匹配"] + names if names else ["自动匹配", "未找到模型"]


def _display_model_name(path: Path) -> str:
    return path.name


def _mmproj_files() -> list[str]:
    files: list[str] = []
    for root in _roots():
        files.extend(str(path) for path in root.rglob("*mmproj*.gguf") if path.is_file())
    return sorted(files, key=lambda item: Path(item).name.lower())


def _mmproj_choices() -> list[str]:
    return ["自动匹配"] + [Path(item).name for item in _mmproj_files()]


def _resolve_choice(choice: str, paths: list[str], label: str) -> Path:
    matches = [Path(item) for item in paths if Path(item).name == choice]
    if not matches:
        raise ValueError(f"请选择有效的{label}。")
    if len(matches) > 1:
        raise ValueError(f"发现同名{label}“{choice}”，请移除重复文件后重启 ComfyUI。")
    return matches[0]


def _resolve_model(choice: str, use_ninfer: bool = False) -> Path:
    if choice == "自动匹配":
        candidates = [Path(item) for item in _scan_models()]
        if use_ninfer:
            candidates = [item for item in candidates if item.suffix.lower() == ".ninfer"]
            _bundled_ninfer_profile()
            preferred = [item for item in candidates if item.name == "Ternary-Bonsai-2-27B.ninfer"]
        else:
            candidates = [item for item in candidates if item.suffix.lower() == ".gguf"]
            preferred = [item for item in candidates if "qwen3.5-9b.q4" in item.name.lower()]
        if preferred:
            return preferred[0]
        if candidates:
            return candidates[0]
        raise ValueError("自动匹配未找到可用模型。")
    matches = [Path(item) for item in _scan_models() if _display_model_name(Path(item)) == choice]
    if not matches:
        raise ValueError("请选择有效的模型。")
    if len(matches) > 1:
        raise ValueError(f"发现同名模型“{choice}”，请移除重复文件后重启 ComfyUI。")
    return matches[0]


def _resolve_mmproj(choice: str, model: Path) -> Path | None:
    if choice == "自动匹配":
        return _find_mmproj(model)
    return _resolve_choice(choice, _mmproj_files(), "mmproj")


def _json_request(url: str, body: dict[str, Any], timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"推理服务返回 HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接推理服务：{exc.reason}") from exc
    except (ConnectionError, OSError, http.client.HTTPException) as exc:
        raise RuntimeError(f"推理服务连接被中断：{exc}") from exc


def _user_requests_reasoning(text: str) -> bool:
    """Only expose reasoning when the user's own request explicitly asks for it."""
    return bool(re.search(
        r"(?:输出|显示|展示|保留|给出).{0,8}(?:分析过程|推理过程|思考过程|思维链|reasoning|analysis)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ))


def _clean_answer(value: Any, allow_reasoning: bool = False) -> str:
    """Return only the final answer unless reasoning was explicitly requested."""
    answer = str(value or "").strip()
    if allow_reasoning:
        return answer
    # Cover the tags emitted by Qwen, llama.cpp and several common chat templates.
    answer = re.sub(
        r"<(?:think|thinking|analysis|reasoning)>.*?</(?:think|thinking|analysis|reasoning)>\s*",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    # Some templates return an untagged analysis section followed by an explicit
    # final-answer marker. In that case retain everything after the final marker.
    final_markers = list(re.finditer(
        r"(?:^|\n)\s*(?:#{1,6}\s*)?(?:最终答案|最终提示词|答案|Final Answer|Final Prompt)\s*[:：]?\s*",
        answer,
        flags=re.IGNORECASE,
    ))
    if final_markers:
        answer = answer[final_markers[-1].end():].strip()
    # Remove harmless wrappers that are not part of the requested prompt text.
    answer = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", answer, flags=re.IGNORECASE).strip()
    return answer


def _server_model_id(api_base: str, fallback: str) -> str:
    """NInfer exposes its artifact model id; use it instead of the local filename."""
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        model_id = data.get("data", [{}])[0].get("id")
        return str(model_id) if model_id else fallback
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return fallback


_OWNED_SERVERS: dict[str, subprocess.Popen[str]] = {}
_OWNED_SERVER_SPECS: dict[str, tuple[str, bool, bool, str, int]] = {}
_LLAMA_CPP_MODELS: dict[tuple[str, str, bool, int], Any] = {}


def _server_alive(api_base: str, attempts: int = 1, retry_delay: float = 0.2) -> bool:
    """Tolerate transient Windows socket resets during engine startup/reload."""
    for attempt in range(max(1, attempts)):
        try:
            with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=2):
                return True
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ConnectionError,
            OSError,
            http.client.HTTPException,
        ):
            if attempt + 1 < attempts:
                time.sleep(retry_delay)
    return False


def _orphaned_bundled_ninfer(api_base: str) -> Any | None:
    """Return a prior-plugin NInfer listener, never a user-owned service.

    ComfyUI restarts do not automatically end child processes.  A leftover plugin
    server may have been started without --vision, so merely reusing it makes an
    IMAGE socket look ignored.  The executable path check is intentionally strict:
    only a process launched from this plugin's engine directory may be replaced.
    """
    endpoint = urlsplit(api_base)
    if endpoint.hostname not in ("127.0.0.1", "localhost") or not endpoint.port:
        return None
    try:
        import psutil

        engine_root = (PLUGIN_DIR / "engine").resolve()
        for connection in psutil.net_connections(kind="tcp"):
            if connection.status != psutil.CONN_LISTEN or connection.laddr.port != endpoint.port or not connection.pid:
                continue
            process = psutil.Process(connection.pid)
            executable = Path(process.exe()).resolve()
            if executable.name.lower() == "ninfer-serve.exe" and engine_root in executable.parents:
                return process
    except (ImportError, OSError, ValueError):
        return None
    return None


def _orphaned_ninfer_matches(process: Any, model: Path, vision: bool, thinking: bool, context_tokens: int) -> bool:
    """Whether a prior-plugin process already has all capabilities this request needs."""
    try:
        arguments = [str(item) for item in process.cmdline()]
        lowered = [item.lower() for item in arguments]
        if str(model).lower() not in lowered:
            return False
        # A vision-enabled server is also valid for a text-only request.
        if vision and "--vision" not in lowered:
            return False
        if thinking and "--no-thinking" in lowered:
            return False
        if not thinking and "--no-thinking" not in lowered:
            return False
        context_index = lowered.index("--max-context")
        return int(arguments[context_index + 1]) >= context_tokens
    except (AttributeError, ValueError, IndexError):
        return False


def _process_is_running(process: Any) -> bool:
    """Support both subprocess.Popen and psutil.Process objects."""
    try:
        if hasattr(process, "poll"):
            return process.poll() is None
        return bool(process.is_running())
    except Exception:
        return False


def _bundled_ninfer_profile() -> dict[str, Any]:
    """Return the packaged NInfer profile for this GPU; never search user disks for an engine."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        capability = float(output.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
        raise RuntimeError("启用 NInfer 需要受支持的 NVIDIA 显卡及 nvidia-smi。") from exc

    if capability >= 12.0:
        profile = {"id": "sm120", "engine_dir": "ninfer-sm120", "kv_dtype": "fp8", "max_context": AUTO_CONTEXT_MAX, "vision_context": AUTO_CONTEXT_MAX}
    elif capability >= 8.9:
        profile = {"id": "sm89", "engine_dir": "ninfer-sm89", "kv_dtype": "fp8", "max_context": AUTO_CONTEXT_MAX, "vision_context": AUTO_CONTEXT_MAX}
    else:
        raise RuntimeError("内置 NInfer 仅支持 40 系与 50 系显卡。其他显卡请关闭 NInfer，改用 GGUF 模型。")

    engine = PLUGIN_DIR / "engine" / profile["engine_dir"] / "ninfer-serve.exe"
    if not engine.is_file():
        raise RuntimeError("当前显卡不受内置 NInfer 引擎支持，或插件 engine 文件不完整。")
    profile["engine"] = engine
    return profile


def _bundled_ninfer_engine() -> Path:
    return _bundled_ninfer_profile()["engine"]


def _effective_ninfer_context(vision: bool, context_tokens: int) -> int:
    """Return the actual capacity passed to the selected bundled engine."""
    profile = _bundled_ninfer_profile()
    limit = profile["vision_context"] if vision else profile["max_context"]
    return min(max(AUTO_CONTEXT_MIN, int(context_tokens)), limit)


def _bundled_ninfer_environment() -> dict[str, str]:
    runtime = PLUGIN_DIR / "engine" / "runtime"
    if not (runtime / "VCRUNTIME140.dll").is_file() or not (runtime / "MSVCP140.dll").is_file():
        raise RuntimeError("内置 NInfer 运行库不完整：请重新安装插件 engine/runtime 文件。")
    environment = os.environ.copy()
    # Shared 40/50-series dependencies shipped by this plugin.
    engine_dir = _bundled_ninfer_profile()["engine"].parent
    environment["PATH"] = str(engine_dir) + os.pathsep + str(runtime) + os.pathsep + environment.get("PATH", "")
    return environment


def _startup_log_tail(kind: str, lines: int = 20) -> str:
    log = PLUGIN_DIR / "engine" / f"{kind}-startup.log"
    try:
        content = log.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:]) or "（引擎没有输出日志）"
    except OSError:
        return "（未生成引擎日志）"


def _ninfer_startup_error(detail: str, timed_out: bool = False) -> RuntimeError:
    """Turn the engine's VRAM planner failure into an actionable node error."""
    match = re.search(r"model weights require (\d+) bytes of device memory, but only (\d+) bytes are free", detail)
    if match:
        required = int(match.group(1)) / (1024 ** 3)
        available = int(match.group(2)) / (1024 ** 3)
        return RuntimeError(
            f"内置 NInfer 无法加载该模型：模型权重需要 {required:.2f} GiB 可用显存，"
            f"当前只有 {available:.2f} GiB。此错误发生在 KV 上下文分配之前，"
            "降低上下文不会解决；请释放显存后重试，或关闭 NInfer 改用 GGUF。"
        )
    reason = "启动超时" if timed_out else "启动后已退出"
    return RuntimeError(f"内置 NInfer 服务{reason}。引擎日志：\n{detail}")


def _bundled_ninfer_command(model: Path, vision: bool, thinking: bool, context_tokens: int) -> list[str]:
    endpoint = urlsplit(_settings()["ninfer_api_base"])
    if endpoint.hostname not in ("127.0.0.1", "localhost") or not endpoint.port:
        raise RuntimeError("内置 NInfer 引擎只支持本机地址。请将 ninfer_api_base 保持为 127.0.0.1:8080/v1。")
    profile = _bundled_ninfer_profile()
    context_tokens = _effective_ninfer_context(vision, context_tokens)
    command = [
        str(profile["engine"]), str(model), "--host", "127.0.0.1", "--port", str(endpoint.port),
        "--max-context", str(context_tokens), "--kv-capacity", str(context_tokens), "--max-concurrency", "1", "--kv-dtype", profile["kv_dtype"],
        "--default-max-tokens", str(context_tokens), "--wddm-evictable-budget",
    ]
    if vision:
        command.append("--vision")
    command.append("--preserve-thinking" if thinking else "--no-thinking")
    return command


def _ensure_server(kind: str, model: Path, vision: bool, mmproj: Path | None = None, thinking: bool = True, context_tokens: int = AUTO_CONTEXT_MIN) -> None:
    """Start only a server created by this plugin. Never terminate an unknown user service."""
    settings = _settings()
    api_base = settings["ninfer_api_base"] if kind == "ninfer" else settings["llama_api_base"]
    # Keep the recorded capacity identical to the command line.  In particular,
    # visual NInfer has a smaller hardware profile cap than text-only NInfer.
    if kind == "ninfer":
        context_tokens = _effective_ninfer_context(vision, context_tokens)
    spec = (str(model), vision, thinking, str(mmproj or ""), context_tokens)
    old = _OWNED_SERVERS.get(kind)
    healthy = _server_alive(api_base, attempts=3)
    orphan = _orphaned_bundled_ninfer(api_base) if kind == "ninfer" else None

    # A process can own/listen on the port while briefly resetting health-check
    # connections.  Inspect the process before deciding to launch another copy.
    # This prevents a transient WinError 10054 from turning into a port conflict.
    existing = old if old and _process_is_running(old) else orphan
    if existing is not None:
        if existing is old:
            old_spec = _OWNED_SERVER_SPECS.get(kind)
            compatible = (
                old_spec is not None
                and old_spec[0] == spec[0]
                and (old_spec[1] or not spec[1])
                and old_spec[2:4] == spec[2:4]
                and old_spec[4] >= spec[4]
            )
        else:
            compatible = _orphaned_ninfer_matches(existing, model, vision, thinking, context_tokens)
        if compatible:
            if healthy:
                return
            # Allow a busy/reloading service to recover without starting or
            # killing another engine.  Repeated connection resets become a
            # controlled diagnostic instead of leaking a raw socket exception.
            recovery_deadline = time.time() + 15
            while time.time() < recovery_deadline and _process_is_running(existing):
                if _server_alive(api_base, attempts=2):
                    return
                time.sleep(0.5)
            if _process_is_running(existing):
                raise RuntimeError(
                    "NInfer 进程仍在运行，但健康检查连接持续被重置。"
                    "请稍后重试；若持续发生，请查看 engine/ninfer-startup.log。"
                )
        if _process_is_running(existing):
            existing.terminate()
            try:
                existing.wait(timeout=10)
            except Exception:
                existing.kill()
                try:
                    existing.wait(timeout=10)
                except Exception:
                    pass
    elif healthy:
        # A healthy service not owned by this plugin is intentionally reused.
        return
    if kind == "ninfer":
        command: str | list[str] = _bundled_ninfer_command(model, vision, thinking, context_tokens)
        shell = False
    else:
        command_template = settings.get("llama_server_command", "").strip()
        if not command_template:
            raise RuntimeError(
                "llama 服务未运行。请先启动服务，或在 config/settings.json 的 "
                "llama_server_command 中配置启动模板。"
            )
        command = command_template.format(model=str(model), mmproj=str(mmproj or ""), vision="--vision" if vision else "")
        shell = True
    if kind != "ninfer" and not command:
        raise RuntimeError(
            f"{kind} 服务未运行。请先启动服务，或在 config/settings.json 的 "
            f"{kind}_server_command 中配置启动模板。"
        )
    if old and old.poll() is None:
        old.terminate()
    log_path = PLUGIN_DIR / "engine" / f"{kind}-startup.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} start ---\n")
        _OWNED_SERVERS[kind] = subprocess.Popen(
            command,
            shell=shell,
            cwd=str(PLUGIN_DIR),
            env=_bundled_ninfer_environment() if kind == "ninfer" else None,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    _OWNED_SERVER_SPECS[kind] = spec
    deadline = time.time() + 120
    while time.time() < deadline:
        if _server_alive(api_base):
            return
        # Do not make the workflow wait a full two minutes for a deterministic
        # load failure such as insufficient VRAM or a missing DLL.
        if _OWNED_SERVERS[kind].poll() is not None:
            detail = _startup_log_tail(kind)
            _OWNED_SERVERS.pop(kind, None)
            _OWNED_SERVER_SPECS.pop(kind, None)
            if kind == "ninfer":
                raise _ninfer_startup_error(detail)
            raise RuntimeError(f"{kind} 服务启动后已退出。引擎日志：\n{detail}")
        time.sleep(0.5)
    detail = _startup_log_tail(kind)
    if kind == "ninfer":
        raise _ninfer_startup_error(detail, timed_out=True)
    raise RuntimeError(f"{kind} 服务启动超时。引擎日志：\n{detail}")


def _unload_plugin_model(kind: str | None) -> None:
    """Release only processes owned by this plugin, then return Python/Torch caches to the OS/driver."""
    if kind:
        process = _OWNED_SERVERS.get(kind)
        # A ComfyUI restart turns an otherwise plugin-owned NInfer process into an
        # orphan.  It is still safe to unload because its executable was verified
        # to live below this plugin's engine directory.
        if process is None and kind == "ninfer":
            process = _orphaned_bundled_ninfer(_settings()["ninfer_api_base"])
        if process and _process_is_running(process):
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()
        _OWNED_SERVERS.pop(kind, None)
        _OWNED_SERVER_SPECS.pop(kind, None)
    for loaded in _LLAMA_CPP_MODELS.values():
        try:
            loaded.close()
        except Exception:
            pass
    _LLAMA_CPP_MODELS.clear()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _release_comfy_vram_for_ninfer() -> None:
    """Return ComfyUI model allocations to CUDA before the near-full-VRAM NInfer load.

    The bundled 27B runtime needs almost the complete 16 GiB device.  Merely calling
    ``torch.cuda.empty_cache`` does not unload ComfyUI's managed diffusion/text models,
    leaving roughly 1-2 GiB occupied and making NInfer's planner fail before KV setup.
    Media tensors have already been encoded to CPU data URLs when this is called.
    """
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        try:
            model_management.soft_empty_cache(True)
        except TypeError:
            model_management.soft_empty_cache()
    except (ImportError, AttributeError):
        pass
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _resize_media_to_budget(
    image_urls: list[str], total_pixels: int, max_item_pixels: int | None = None,
) -> list[str]:
    """Downscale data-URL images to a shared pixel budget without cropping."""
    if not image_urls:
        return []
    per_item = total_pixels // len(image_urls)
    if max_item_pixels is not None:
        per_item = min(per_item, max_item_pixels)
    per_item = max(256 * 256, per_item)
    output: list[str] = []
    for url in image_urls:
        try:
            _, encoded = url.split(",", 1)
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as source:
                image = source.convert("RGB")
                width, height = image.size
                area = width * height
                if area > per_item:
                    scale = (per_item / area) ** 0.5
                    width = max(32, int(width * scale))
                    height = max(32, int(height * scale))
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                stream = io.BytesIO()
                # JPEG keeps the aggregate encoded-media budget predictable and
                # is sufficient for prompt understanding; source files remain untouched.
                image.save(stream, format="JPEG", quality=90, optimize=True)
            output.append("data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii"))
        except (OSError, ValueError, TypeError, base64.binascii.Error) as exc:
            raise ValueError("图像或视频帧不是有效的图像数据，无法自动缩放。") from exc
    return output


def _api_chat(
    api_base: str, model: str, instruction: str, image_urls: list[str], allow_reasoning: bool = False,
) -> str:
    result: dict[str, Any] | None = None
    levels = NINFER_VISION_LEVELS if image_urls else ((0, 0),)
    last_error: RuntimeError | None = None
    for attempt, (total_budget, item_budget) in enumerate(levels):
        resized_urls = (
            _resize_media_to_budget(image_urls, total_budget, item_budget)
            if image_urls else []
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        content.extend({"type": "image_url", "image_url": {"url": item}} for item in resized_urls)
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是提示词助手。默认只输出可直接使用的正向提示词纯文本，不添加解释、标题或 Markdown。"
                        "可以在内部思考，但最终消息严禁包含分析、推理步骤、思考过程或总结；"
                        "只有用户明确要求其他输出格式或展示分析时，才遵循用户的额外约束。"
                        + (
                            "本轮已附带图像或视频画面，必须以媒体中实际可见的像素内容为首要事实，"
                            "不得把模板中的示例描述当成画面内容，也绝不能要求用户再次上传媒体。"
                            if resized_urls else ""
                        )
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": 0.7,
            "stream": False,
        }
        try:
            result = _json_request(api_base.rstrip("/") + "/chat/completions", payload, int(_settings()["request_timeout_seconds"]))
            break
        except RuntimeError as exc:
            last_error = exc
            if "media_budget_exceeded" not in str(exc) or attempt == len(levels) - 1:
                raise
    if result is None:
        raise last_error or RuntimeError("推理服务没有返回结果。")
    try:
        return _clean_answer(result["choices"][0]["message"]["content"], allow_reasoning)
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"推理服务返回了无法识别的内容：{result}") from exc


def _tensor_data_url(tensor: Any, max_pixels: int | None = None) -> str:
    if torch is not None and isinstance(tensor, torch.Tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim == 4:
        raise ValueError("请逐张传入图像，不要将整个批次作为单个动态端口传入。")
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("图像必须是 ComfyUI 的 RGB/RGBA IMAGE。")
    rgb = array[..., :3]
    # Native ComfyUI IMAGE tensors are float [0, 1]; accepting uint8 as well
    # keeps IMAGE outputs from third-party video/image nodes from turning white.
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = rgb * 255.0
    array = np.clip(rgb, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, "RGB")
    # Resize before encoding. This avoids constructing and then decoding a very
    # large PNG only to shrink it again in the shared-budget stage.
    if max_pixels and image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=92, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _frames_from_tensor(images: Any, max_pixels: int = 1024 * 1024) -> list[str]:
    if torch is not None and isinstance(images, torch.Tensor):
        array = images
    else:
        array = np.asarray(images) if images is not None else None
    if array is not None and array.ndim == 4:
        count = array.shape[0]
        indexes = np.linspace(0, count - 1, min(10, count), dtype=int)
        return [_tensor_data_url(array[index], max_pixels) for index in indexes]
    return []


def _video_frame_urls_from_file(video_path: str) -> list[str]:
    path = Path(video_path)
    if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("视频必须是存在的 mp4/mov/webm/mkv/avi 文件路径。")
    ffmpeg = _settings()["ffmpeg"]
    if not shutil.which(ffmpeg) and not Path(ffmpeg).is_file():
        raise RuntimeError("未找到 FFmpeg。请安装 FFmpeg，或在 config/settings.json 的 ffmpeg 中填写 ffmpeg.exe 路径。")

    # Sample across the entire clip, rather than taking only the first 30
    # seconds.  ffprobe ships with normal FFmpeg distributions; a compatible
    # fps fallback remains for minimal FFmpeg installations without it.
    ffprobe_candidates = [str(Path(ffmpeg).with_name("ffprobe.exe"))] if Path(ffmpeg).is_file() else []
    ffprobe_candidates.extend([item for item in (shutil.which("ffprobe"), shutil.which("ffprobe.exe")) if item])
    duration: float | None = None
    for ffprobe in dict.fromkeys(ffprobe_candidates):
        if not Path(ffprobe).is_file() and not shutil.which(ffprobe):
            continue
        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            value = float(probe.stdout.strip())
            if probe.returncode == 0 and np.isfinite(value) and value > 0:
                duration = value
                break
        except (OSError, ValueError, subprocess.TimeoutExpired):
            continue
    with tempfile.TemporaryDirectory(prefix="aimanzi_video_") as directory:
        urls: list[str] = []
        if duration is not None:
            # One frame per three seconds for short clips, capped at ten evenly
            # spread frames for long clips.  No keyframe-count widget is needed.
            count = min(MAX_DYNAMIC_IMAGE_INPUTS, max(1, int(np.ceil(duration / 3))))
            for index in range(count):
                frame = Path(directory) / f"frame_{index:02d}.jpg"
                timestamp = duration * index / count
                command = [
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}", "-i", str(path),
                    "-frames:v", "1", "-vf", "scale='min(1024,iw)':-2", "-q:v", "3", "-y", str(frame),
                ]
                process = subprocess.run(command, capture_output=True, text=True, timeout=45)
                if process.returncode == 0 and frame.is_file() and frame.stat().st_size > 0:
                    urls.append("data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode("ascii"))
        else:
            out_pattern = str(Path(directory) / "frame_%02d.jpg")
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path), "-vf", "fps=1/3,scale='min(1024,iw)':-2", "-frames:v", str(MAX_DYNAMIC_IMAGE_INPUTS), "-q:v", "3", out_pattern]
            process = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if process.returncode != 0:
                raise RuntimeError("FFmpeg 读取视频失败：" + process.stderr.strip())
            for frame in sorted(Path(directory).glob("frame_*.jpg")):
                urls.append("data:image/jpeg;base64," + base64.b64encode(frame.read_bytes()).decode("ascii"))
        if not urls:
            raise RuntimeError("未能从视频提取有效画面。")
        return urls


def _video_frame_urls(video: Any) -> list[str]:
    """Accept ComfyUI VIDEO objects from core/Easy-Media, not a manual path widget."""
    if isinstance(video, str):  # Allows legacy saved workflows to fail gracefully or keep working.
        return _video_frame_urls_from_file(video.strip().strip('"'))
    try:
        components = video.get_components()
        frames = _frames_from_tensor(getattr(components, "images", None))
        if frames:
            return frames
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    try:
        source = video.get_stream_source()
        if isinstance(source, (str, os.PathLike)) and Path(source).is_file():
            return _video_frame_urls_from_file(str(source))
    except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
        pass
    temp_path = ""
    try:
        fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        video.save_to(temp_path)
        return _video_frame_urls_from_file(temp_path)
    except Exception as exc:
        raise RuntimeError("无法读取 VIDEO 输入。请使用标准 VIDEO 输出节点或 ComfyUI-Easy-Media 的视频加载节点。") from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def _find_mmproj(model_path: Path) -> Path | None:
    candidates = sorted(model_path.parent.glob("*mmproj*.gguf"))
    return candidates[0] if candidates else None


def _find_visual_bridge_model() -> tuple[Path, Path]:
    """Choose a local GGUF vision model for grounding NInfer media requests.

    Some converted/ternary NInfer language bodies are not aligned with the vision
    projector shipped in the artifact.  In that case the server accepts the image but
    describes unrelated content.  A small, known multimodal GGUF is therefore used to
    extract visual facts before NInfer performs the final writing pass.
    """
    candidates: list[tuple[int, Path, Path]] = []
    for model_name in _scan_models():
        path = Path(model_name).resolve()
        if path.suffix.lower() != ".gguf" or "mmproj" in path.name.lower():
            continue
        mmproj = _find_mmproj(path)
        if mmproj is None:
            continue
        name = path.name.lower()
        if "qwen" not in name or ("vl" not in name and "3.5" not in name):
            continue
        priority = 0 if "qwen3-vl" in name or "qwen3_vl" in name else 1
        candidates.append((priority, path, mmproj))
    if not candidates:
        raise RuntimeError(
            "NInfer 图像推理需要视觉桥接模型。请在 models/LLM 中放入 Qwen3-VL GGUF "
            "及其匹配的 mmproj*.gguf；纯文字 NInfer 不受影响。"
        )
    _, model, mmproj = sorted(candidates, key=lambda item: (item[0], item[1].name.lower()))[0]
    return model, mmproj


def _ninfer_visual_facts(instruction: str, image_urls: list[str]) -> str:
    bridge_model, bridge_mmproj = _find_visual_bridge_model()
    grounded_media = _resize_media_to_budget(
        image_urls,
        total_pixels=4 * 1024 * 1024,
        max_item_pixels=1024 * 1024,
    )
    observation_request = (
        "只观察并准确记录输入图像或视频帧中实际可见的内容。详尽说明主体身份与数量、外观、动作、"
        "环境、物体、构图、镜头、光线、颜色、材质、风格以及可辨识文字。不要生成提示词，不要猜测"
        "未出现的对象，不要把下面的任务文字或模板示例当作画面内容。\n\n"
        "用户最终任务（仅用于确定观察重点）：\n" + instruction
    )
    facts = _llamacpp_chat(bridge_model, observation_request, grounded_media, bridge_mmproj, False)
    if not facts.strip():
        raise RuntimeError("视觉桥接模型没有返回可用的图像/视频识别结果。")
    return facts


def _validate_model(model: str, use_ninfer: bool, has_media: bool, selected_mmproj: str) -> tuple[Path, Path | None]:
    path = _resolve_model(model, use_ninfer)
    suffix = path.suffix.lower()
    if use_ninfer and suffix != ".ninfer":
        raise ValueError("启用 NInfer 时只能选择 .ninfer 模型。")
    if not use_ninfer and suffix != ".gguf":
        raise ValueError("关闭 NInfer 后请选择 .gguf 模型。")
    mmproj: Path | None = None
    if has_media and not use_ninfer:
        if path.suffix.lower() == ".gguf":
            mmproj = _resolve_mmproj(selected_mmproj, path)
        if path.suffix.lower() == ".gguf" and mmproj is None:
            raise ValueError("视觉 GGUF 模型缺少同目录 mmproj*.gguf，不能处理图像或视频。")
    return path, mmproj


def _llamacpp_chat(
    model_path: Path,
    instruction: str,
    image_urls: list[str],
    mmproj: Path | None,
    thinking: bool,
    context_tokens: int | None = None,
    allow_reasoning: bool = False,
) -> str:
    """Universal GGUF backend using ComfyUI's bundled llama-cpp-python CPU runtime.

    It needs no separately installed llama-server. The current ComfyUI wheel is CPU-only, so it
    works on all Windows devices; a CUDA/Vulkan wheel can later replace it without workflow edits.
    """
    # ComfyUI/Torch can preload Intel OpenMP while the installed llama-cpp-python
    # wheel loads LLVM OpenMP.  Without this compatibility flag Windows aborts with
    # OMP Error #15 before any GGUF request can run.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen35ChatHandler
    except ImportError as exc:
        raise RuntimeError("关闭 NInfer 的 GGUF 推理需要 ComfyUI Python 中的 llama-cpp-python。") from exc
    context_tokens = context_tokens or _auto_context_tokens(instruction, len(image_urls))
    key = (str(model_path), str(mmproj or ""), thinking, context_tokens)
    llm = _LLAMA_CPP_MODELS.get(key)
    if llm is None:
        handler = None
        if image_urls:
            if mmproj is None:
                raise RuntimeError("图像或视频 GGUF 推理需要匹配的 mmproj 文件。")
            lowered = model_path.name.lower()
            if "qwen3.5" in lowered:
                handler = Qwen35ChatHandler(
                    clip_model_path=str(mmproj),
                    enable_thinking=thinking,
                    use_gpu=False,
                    verbose=False,
                    image_min_tokens=1024,
                )
            elif "qwen" in lowered and "vl" in lowered:
                handler = Qwen3VLChatHandler(
                    clip_model_path=str(mmproj),
                    force_reasoning=thinking,
                    use_gpu=False,
                    verbose=False,
                    image_min_tokens=1024,
                )
            else:
                raise RuntimeError("当前仅内置 Qwen3-VL 与 Qwen3.5 GGUF 的视觉处理器。")
        try:
            llm = Llama(
                model_path=str(model_path),
                chat_handler=handler,
                n_ctx=context_tokens,
                n_gpu_layers=0,
                n_threads=max(1, (os.cpu_count() or 4) - 1),
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"GGUF 模型加载失败：{exc}") from exc
        _LLAMA_CPP_MODELS[key] = llm
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    content.extend({"type": "image_url", "image_url": {"url": item}} for item in image_urls)
    try:
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是提示词助手。默认只输出可直接使用的正向提示词纯文本，不添加解释、标题或 Markdown。"
                        "可以在内部思考，但最终消息严禁包含分析、推理步骤、思考过程或总结；"
                        "只有用户明确要求其他输出格式或展示分析时，才遵循用户的额外约束。"
                        + (
                            "本轮已附带图像或视频画面，必须以媒体中实际可见的像素内容为首要事实，"
                            "不得把模板中的示例描述当成画面内容，也绝不能要求用户再次上传媒体。"
                            if image_urls else ""
                        )
                    ),
                },
                {"role": "user", "content": content if image_urls else instruction},
            ],
            temperature=0.7,
            max_tokens=None,
        )
        return _clean_answer(result["choices"][0]["message"]["content"], allow_reasoning)
    except ValueError as exc:
        message = str(exc)
        match = re.search(r"Requested tokens \((\d+)\) exceed context window", message)
        if match:
            required_tokens = int(match.group(1))
            next_context = _auto_context_tokens(instruction, len(image_urls), required_tokens)
            if next_context > context_tokens:
                # The exact chat template can be longer than the pre-load estimate. Reload once at
                # the required size rather than forcing users to choose a context manually.
                loaded = _LLAMA_CPP_MODELS.pop(key, None)
                try:
                    if loaded is not None:
                        loaded.close()
                except Exception:
                    pass
                return _llamacpp_chat(
                    model_path, instruction, image_urls, mmproj, thinking, next_context, allow_reasoning
                )
            raise RuntimeError(
                f"GGUF 输入需要至少 {required_tokens} tokens，已达到模型/硬件自动上下文上限 "
                f"{AUTO_CONTEXT_MAX} tokens。请缩短 TXT/MD 模板、减少图片或关闭思考模式后重试。"
            ) from exc
        raise RuntimeError(f"GGUF 推理失败：{exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"GGUF 推理失败：{exc}") from exc


def _template_files() -> list[str]:
    # Kept for backward-compatible imports. The node itself deliberately uses STRING instead
    # of a COMBO so ComfyUI does not reject an already-uploaded/legacy filename at validation.
    return ["点击上传 TXT/MD 文件"]


def _decode_template_text(data: bytes, label: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文本编码：{label}。请使用 UTF-8 或 GB18030。")


def _safe_skill_member(info: zipfile.ZipInfo) -> PurePosixPath:
    raw = info.filename.replace("\\", "/")
    member = PurePosixPath(raw)
    if (
        not raw
        or raw.startswith("/")
        or re.match(r"^[A-Za-z]:", raw)
        or any(part in ("", ".", "..") for part in member.parts)
    ):
        raise ValueError(f"Skill 包包含不安全路径：{info.filename}")
    # Unix symlinks can point outside the archive even when their member name is safe.
    if ((info.external_attr >> 16) & 0o170000) == 0o120000:
        raise ValueError(f"Skill 包不允许符号链接：{info.filename}")
    return member


def _skill_frontmatter(skill_text: str) -> tuple[str, str]:
    if not skill_text.startswith("---"):
        return "", ""
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---(?:\s*\r?\n|$)", skill_text, re.DOTALL)
    if not match:
        return "", ""
    header = match.group(1)
    name_match = re.search(r"(?m)^name\s*:\s*[\"']?([^\r\n\"']+)", header)
    description_match = re.search(r"(?m)^description\s*:\s*[\"']?([^\r\n\"']+)", header)
    return (
        name_match.group(1).strip() if name_match else "",
        description_match.group(1).strip() if description_match else "",
    )


def _read_skill_archive(path: Path) -> str:
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Skill 文件不是有效的 ZIP/.skill 压缩包。") from exc
    with archive:
        infos = [item for item in archive.infolist() if not item.is_dir()]
        if len(infos) > SKILL_MAX_FILES:
            raise ValueError(f"Skill 包文件过多（{len(infos)} 个），安全上限为 {SKILL_MAX_FILES} 个。")
        total_size = sum(item.file_size for item in infos)
        if total_size > SKILL_MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Skill 包解压后超过 64 MiB 安全上限。")
        safe_members: dict[str, zipfile.ZipInfo] = {}
        for info in infos:
            member = _safe_skill_member(info)
            if info.flag_bits & 0x1:
                raise ValueError("不支持加密的 Skill 压缩包。")
            if info.compress_size and info.file_size > max(1024 * 1024, info.compress_size * 200):
                raise ValueError(f"Skill 包包含异常压缩比文件：{info.filename}")
            safe_members[member.as_posix()] = info

        entry_names = [name for name in safe_members if PurePosixPath(name).name.lower() == "skill.md"]
        if not entry_names:
            raise ValueError("压缩包内未找到 SKILL.md，不能作为 Skill 使用。")
        if len(entry_names) > 1:
            raise ValueError("压缩包内存在多个 SKILL.md，请将每个 Skill 分别打包上传。")
        entry_name = entry_names[0]
        entry_info = safe_members[entry_name]
        if entry_info.file_size > SKILL_MAX_TEXT_FILE_BYTES:
            raise ValueError("SKILL.md 超过 1 MiB 安全上限。")
        skill_text = _decode_template_text(archive.read(entry_info), entry_name)
        skill_root = PurePosixPath(entry_name).parent
        name, description = _skill_frontmatter(skill_text)

        references: list[tuple[str, str]] = []
        text_bytes = len(skill_text.encode("utf-8"))
        for member_name, info in sorted(safe_members.items()):
            member = PurePosixPath(member_name)
            if member_name == entry_name or info.file_size == 0:
                continue
            try:
                relative = member.relative_to(skill_root)
            except ValueError:
                continue
            # Read documentation resources only. Scripts and assets are never executed
            # or injected, even if their extension happens to contain readable text.
            if not relative.parts or relative.parts[0].lower() != "references":
                continue
            if member.suffix.lower() not in SKILL_TEXT_EXTENSIONS:
                continue
            if info.file_size > SKILL_MAX_TEXT_FILE_BYTES:
                continue
            data = archive.read(info)
            if text_bytes + len(data) > SKILL_MAX_TEXT_BYTES:
                break
            content = _decode_template_text(data, member_name)
            references.append((relative.as_posix(), content))
            text_bytes += len(content.encode("utf-8"))

    sections = [
        "[Skill 元数据]",
        f"名称：{name or PurePosixPath(entry_name).parent.name or path.stem}",
        f"描述：{description or '未提供'}",
        f"入口：{entry_name}",
        "",
        "[Skill 核心指令]",
        skill_text,
    ]
    for reference_name, content in references:
        sections.extend(("", f"[Skill 参考资料：{reference_name}]", content))
    return "\n".join(sections).strip()


def _read_template_or_skill(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        if path.stat().st_size > SKILL_MAX_TEXT_BYTES:
            raise ValueError("模板文件超过 8 MiB 安全上限。")
        return _decode_template_text(path.read_bytes(), path.name)
    if suffix in {".skill", ".zip"}:
        return _read_skill_archive(path)
    raise ValueError("仅支持 txt、md、markdown、skill 或包含 SKILL.md 的 zip 文件。")


class AIManziMultimodalPrompt:
    CATEGORY = "AI蛮子/提示词"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("out",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Keep the user prompt at the top of the node. All runtime/model knobs follow it.
                "文字要求": ("STRING", {"multiline": True, "default": "根据输入内容生成可直接使用的正向提示词。"}),
                "启用_ninfer": ("BOOLEAN", {"default": True}),
                "启用思考模式": ("BOOLEAN", {"default": True}),
                "推理后卸载模型": ("BOOLEAN", {"default": False}),
                "模型": (_model_choices(),),
                "mmproj": (_mmproj_choices(),),
            },
            # The frontend reveals the next image socket after the preceding
            # one is connected.  This special mapping makes every revealed
            # 图像_1 ... 图像_10 socket a real, validated ComfyUI IMAGE input.
            "optional": _DynamicImageOptionalInputs({
                "模板输入": ("AIMANZI_TEMPLATE",),
                "图像_1": ("IMAGE",),
                "视频输入": ("VIDEO",),
            }),
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def generate(self, **kwargs):
        use_ninfer = bool(kwargs["启用_ninfer"])
        thinking = bool(kwargs.get("启用思考模式", True))
        unload_after = bool(kwargs.get("推理后卸载模型", False))
        instruction = str(kwargs.get("文字要求", "")).strip()
        # Decide this from the user's own text before appending templates or
        # machine-generated visual facts, so a Skill cannot expose chain-of-thought.
        allow_reasoning = _user_requests_reasoning(instruction)
        source_text = str(kwargs.get("模板输入", "")).strip()
        if source_text:
            instruction += (
                "\n\n以下是用户提供的模板/Skill 指令。将其作为提示词生成规则使用；"
                "其中的示例不得覆盖输入图像、视频或用户明确要求：\n" + source_text
            )
        image_urls: list[str] = []
        # NInfer media is grounded by the Qwen bridge at about 1MP per item.
        # Apply that cap while converting the Comfy tensor, not after creating
        # a full-resolution PNG/data URL. GGUF also gets a conservative 2MP
        # pre-cap before the later shared 6MP budget is enforced.
        input_item_pixels = 1024 * 1024 if use_ninfer else 2 * 1024 * 1024
        for key in sorted(
            (item for item in kwargs if re.fullmatch(r"图像_\d+", item) and int(item.split("_")[-1]) <= MAX_DYNAMIC_IMAGE_INPUTS),
            key=lambda item: int(item.split("_")[-1]),
        ):
            value = kwargs.get(key)
            if value is None:
                continue
            # Comfy IMAGE may contain a batch; retain all batch images in socket order.
            if (torch is not None and isinstance(value, torch.Tensor) and value.ndim == 4) or (
                isinstance(value, np.ndarray) and value.ndim == 4
            ):
                image_urls.extend(_tensor_data_url(value[index], input_item_pixels) for index in range(value.shape[0]))
            else:
                image_urls.append(_tensor_data_url(value, input_item_pixels))
        video = kwargs.get("视频输入")
        if video is not None:
            image_urls.extend(_video_frame_urls(video))
        model, mmproj = _validate_model(str(kwargs["模型"]), use_ninfer, bool(image_urls), str(kwargs["mmproj"]))
        backend_kind: str | None = None
        try:
            if use_ninfer:
                backend_kind = "ninfer"
                # Direct vision in some ternary NInfer artifacts is structurally present but
                # semantically misaligned with the converted language body. Ground the media
                # with a local Qwen vision GGUF, then let NInfer do the final writing pass.
                if image_urls:
                    visual_facts = _ninfer_visual_facts(instruction, image_urls)
                    instruction += (
                        "\n\n以下内容由视觉桥接模型从本次输入图像/视频像素中提取，"
                        "是生成结果时必须遵守的画面事实。不得声称未收到媒体，也不得用模板示例覆盖这些事实：\n"
                        + visual_facts
                    )
                    image_urls = []
                    # The bridge has finished and its result is plain text. Close its
                    # llama.cpp instance before NInfer reserves host/device memory.
                    _unload_plugin_model(None)
                # ComfyUI intentionally keeps recently used diffusion/CLIP/VAE models
                # resident. NInfer's 27B loader requires nearly all of a 16 GiB GPU,
                # so release those managed allocations before starting the engine.
                _release_comfy_vram_for_ninfer()
                context_tokens = _auto_context_tokens(instruction, 0)
                _ensure_server("ninfer", model, False, thinking=thinking, context_tokens=context_tokens)
                api_base = _settings()["ninfer_api_base"]
                answer = _api_chat(
                    api_base,
                    _server_model_id(api_base, model.name),
                    instruction,
                    [],
                    allow_reasoning,
                )
            else:
                backend_kind = "llama_cpp"
                image_urls = _resize_media_to_budget(image_urls, VISION_SAFE_TOTAL_PIXELS)
                answer = _llamacpp_chat(
                    model, instruction, image_urls, mmproj, thinking, allow_reasoning=allow_reasoning
                )
        finally:
            if unload_after:
                _unload_plugin_model(backend_kind)
        if not answer:
            raise RuntimeError("模型没有返回正向提示词。")
        return (answer,)


class AIManziReadText:
    CATEGORY = "AI蛮子/提示词"
    RETURN_TYPES = ("AIMANZI_TEMPLATE",)
    RETURN_NAMES = ("模板输入",)
    FUNCTION = "read"

    @classmethod
    def INPUT_TYPES(cls):
        # The browser upload control is supplied by web/js/aimanzi_prompt_ui.js.
        # Do not use image_upload here: ComfyUI then filters the chooser to image formats.
        # Keep the serialized key for compatibility with existing workflows; the
        # node title and upload button communicate the expanded Skill support.
        return {"required": {"TXT/MD 文件": ("STRING", {"default": ""})}}

    def read(self, **kwargs):
        # Accept the former widget key so saved workflows continue to execute.
        selected_file = str(kwargs.get("模板/Skill 文件", kwargs.get("TXT/MD 文件", "")))
        if not selected_file.strip() or selected_file == "点击上传 TXT/MD 文件" or folder_paths is None:
            raise ValueError("请使用节点上的上传按钮选择模板或 Skill 文件。")
        root = Path(folder_paths.get_input_directory()).resolve()
        path = (root / selected_file).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("模板/Skill 文件无效或不在 ComfyUI/input 目录中。")
        return (_read_template_or_skill(path),)


NODE_CLASS_MAPPINGS = {
    "AIManziMultimodalPrompt": AIManziMultimodalPrompt,
    "AIManziReadText": AIManziReadText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIManziMultimodalPrompt": "AI蛮子 多模态提示词工作台",
    "AIManziReadText": "AI蛮子 加载模板 / Skill",
}
