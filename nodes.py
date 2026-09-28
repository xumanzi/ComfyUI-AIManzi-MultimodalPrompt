"""AI蛮子多模态提示词节点。

界面只暴露模型、NInfer 开关、文字和媒体。推理服务路径等一次性配置放在
config/settings.json，避免把硬件/服务参数塞进工作流。
"""
from __future__ import annotations

import base64
import ctypes
import functools
import gc
import hashlib
import http.client
import io
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import OrderedDict
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
    from aiohttp import web
    from server import PromptServer
except ImportError:
    web = None
    PromptServer = None

try:
    import torch
except ImportError:  # ComfyUI always provides torch; keeps module error readable.
    torch = None


PLUGIN_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = PLUGIN_DIR / "config" / "settings.json"
MODEL_EXTENSIONS = {".ninfer", ".gguf"}
AUTO_CONTEXT_MIN = 4096
AUTO_CONTEXT_MAX = 262144
MAX_DYNAMIC_IMAGE_INPUTS = 10
VIDEO_ANALYSIS_LIMITS = {
    "快速（最多16帧）": 16,
    "标准（最多64帧）": 64,
    "高精度（最多128帧）": 128,
    "完整逐帧（最多256帧）": 256,
}
VIDEO_SEGMENT_FRAMES = 8
VISUAL_FACT_CACHE_ITEMS = 64
FINAL_OUTPUT_CACHE_ITEMS = 32
SEED_MAX = 0xFFFFFFFF
INFERENCE_STRATEGIES = ("普通推理", "创新推理")
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
    default_llm_root = (
        Path(folder_paths.models_dir) / "LLM"
        if folder_paths is not None and getattr(folder_paths, "models_dir", None)
        else Path.cwd() / "models" / "LLM"
    )
    defaults: dict[str, Any] = {
        "llm_roots": [str(default_llm_root)],
        "ninfer_api_base": "http://127.0.0.1:8080/v1",
        "request_timeout_seconds": 600,
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
        # UTF-8 byte length badly overestimates Chinese (three bytes usually map to
        # roughly one token), causing harmless templates to jump to a 64K/128K KV
        # allocation. Count CJK and non-CJK separately and retain a safety margin;
        # the exact-token overflow path below still grows and reloads once if needed.
        cjk_count = len(re.findall(r"[\u3400-\u9fff\uf900-\ufaff]", text))
        non_cjk_count = max(0, len(text) - cjk_count)
        estimated_input = int((cjk_count * 1.15) + (non_cjk_count / 3.2) + 256) + image_count * 2048
    else:
        estimated_input = required_tokens
    # Reserve useful generation room without assuming that every long input also
    # needs an equally long answer. max_tokens=None can still consume all remaining
    # context, and the overflow retry protects unusual chat templates.
    output_reserve = max(AUTO_CONTEXT_MIN, min(16384, estimated_input // 2))
    needed = estimated_input + output_reserve + 512
    context = AUTO_CONTEXT_MIN
    while context < needed and context < AUTO_CONTEXT_MAX:
        context *= 2
    return min(context, AUTO_CONTEXT_MAX)


def _ninfer_context_tokens(text: str, thinking: bool) -> int:
    """Reserve enough room for NInfer reasoning to reach a visible final answer."""
    context = _auto_context_tokens(text, 0)
    return max(context, 16384 if thinking else AUTO_CONTEXT_MIN)


def _roots() -> list[Path]:
    result: list[Path] = []
    for item in _settings()["llm_roots"]:
        path = Path(item)
        if path.exists():
            result.append(path)
    return result


@functools.lru_cache(maxsize=1)
def _scan_models() -> list[str]:
    models: list[str] = []
    for root in _roots():
        for path in root.rglob("*"):
            if (path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
                    and not path.name.lower().startswith("mmproj")):
                models.append(str(path))
    return sorted(models, key=lambda p: (Path(p).suffix.lower(), Path(p).name.lower())) or ["未找到模型"]


def _model_choices() -> list[str]:
    """The canvas shows only filenames; execution resolves them to an LLM-root path."""
    names = [_display_model_name(Path(item)) for item in _scan_models()]
    # Keep this legacy sentinel in the schema so old workflows with a value displaced from
    # the former mmproj widget pass ComfyUI's pre-execution combo validation.
    return ["自动匹配"] + names if names else ["自动匹配", "未找到模型"]


def _display_model_name(path: Path) -> str:
    return path.name


@functools.lru_cache(maxsize=1)
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


def _json_request(
    url: str, body: dict[str, Any], timeout: int, headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json"}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    finished = threading.Event()
    result: dict[str, Any] = {}
    failure: list[BaseException] = []

    def send() -> None:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result.update(json.loads(response.read().decode("utf-8")))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            failure.append(RuntimeError(f"推理服务返回 HTTP {exc.code}: {detail}"))
        except urllib.error.URLError as exc:
            failure.append(RuntimeError(f"无法连接推理服务：{exc.reason}"))
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            failure.append(RuntimeError(f"推理服务连接被中断：{exc}"))
        except BaseException as exc:
            failure.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(target=send, name="AIManzi-HTTP", daemon=True)
    worker.start()
    while not finished.wait(0.1):
        if _processing_interrupted():
            endpoint = urlsplit(url)
            if endpoint.hostname in {"127.0.0.1", "localhost", "::1"}:
                # Stopping a plugin-owned local engine cancels its in-flight generation
                # and also releases VRAM. Never terminate a user-owned external server.
                for kind, process in list(_OWNED_SERVERS.items()):
                    if _process_is_running(process):
                        try:
                            process.terminate()
                        except OSError:
                            pass
                        _OWNED_SERVERS.pop(kind, None)
                        _OWNED_SERVER_SPECS.pop(kind, None)
                        _SERVER_MODEL_IDS.clear()
            _throw_if_interrupted()
    _throw_if_interrupted()
    if failure:
        raise failure[0]
    return result


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


def _inference_profile(strategy: str, observation_mode: bool = False) -> tuple[float, float, str]:
    """Return sampling and instruction policy without exposing internal reasoning."""
    if observation_mode:
        return 0.2, 0.8, (
            "只记录媒体中实际可见的事实，保持客观和低随机性；不要补写画面中不存在的内容。"
        )
    if strategy == "创新推理":
        return 0.9, 0.95, (
            "在完整保留用户硬性要求、模板规则和媒体事实的前提下，主动进行合理的创意联想与拓展。"
            "可增强主体细节、环境、构图、镜头、光线、色彩、材质、氛围、叙事感和艺术风格，"
            "但不得改变主体身份与数量、指定文字、关键动作或其他明确约束。"
        )
    return 0.55, 0.85, (
        "严格依据用户输入、模板规则和媒体事实执行，不主动添加未经要求的主体、情节或物体；"
        "优先保证准确、稳定和可控。"
    )


def _creative_seed_instruction(strategy: str, observation_mode: bool, seed: int) -> str:
    if observation_mode or strategy != "创新推理":
        return ""
    return (
        f" 本轮创意变化编号为 {int(seed) & SEED_MAX}。在不违反用户指定格式、模板硬性规则和媒体事实的前提下，"
        "使用该编号驱动本轮独立的创意路径，主动改变细节选择、描述顺序、构图重点、镜头表达和氛围组织；"
        "不要机械复用上一轮的句式与结构。相同编号应尽量保持可复现。"
    )


def _server_model_id(api_base: str, fallback: str) -> str:
    """NInfer exposes its artifact model id; use it instead of the local filename."""
    cached = _SERVER_MODEL_IDS.get(api_base)
    if cached:
        return cached
    try:
        with urllib.request.urlopen(api_base.rstrip("/") + "/models", timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
        model_id = data.get("data", [{}])[0].get("id")
        resolved = str(model_id) if model_id else fallback
        _SERVER_MODEL_IDS[api_base] = resolved
        return resolved
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return fallback


_OWNED_SERVERS: dict[str, subprocess.Popen[str]] = {}
_OWNED_SERVER_SPECS: dict[str, tuple[str, bool, bool, str, int]] = {}
_SERVER_MODEL_IDS: dict[str, str] = {}
_LLAMA_CPP_MODELS: dict[tuple[str, str, bool, int, str, int], Any] = {}
_VISUAL_FACT_CACHE: OrderedDict[str, str] = OrderedDict()
_VISUAL_FACT_CACHE_LOCK = threading.Lock()
_FINAL_OUTPUT_CACHE: OrderedDict[str, str] = OrderedDict()
_FINAL_OUTPUT_CACHE_LOCK = threading.Lock()
_DLL_DIRECTORY_HANDLES: list[Any] = []
_LLAMA_BACKEND_STATE: dict[str, Any] | None = None


def _path_identity(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        stat = path.stat()
        return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        return str(path)


def _visual_cache_key(namespace: str, media_urls: list[str], *parts: Any) -> str:
    digest = hashlib.sha256()
    digest.update(namespace.encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    for url in media_urls:
        digest.update(b"\1")
        digest.update(url.encode("ascii", errors="strict"))
    return digest.hexdigest()


def _visual_cache_get(key: str) -> str | None:
    with _VISUAL_FACT_CACHE_LOCK:
        value = _VISUAL_FACT_CACHE.pop(key, None)
        if value is not None:
            _VISUAL_FACT_CACHE[key] = value
        return value


def _visual_cache_put(key: str, value: str) -> None:
    with _VISUAL_FACT_CACHE_LOCK:
        _VISUAL_FACT_CACHE.pop(key, None)
        _VISUAL_FACT_CACHE[key] = value
        while len(_VISUAL_FACT_CACHE) > VISUAL_FACT_CACHE_ITEMS:
            _VISUAL_FACT_CACHE.popitem(last=False)


def _final_cache_get(key: str) -> str | None:
    with _FINAL_OUTPUT_CACHE_LOCK:
        value = _FINAL_OUTPUT_CACHE.pop(key, None)
        if value is not None:
            _FINAL_OUTPUT_CACHE[key] = value
        return value


def _final_cache_put(key: str, value: str) -> None:
    with _FINAL_OUTPUT_CACHE_LOCK:
        _FINAL_OUTPUT_CACHE.pop(key, None)
        _FINAL_OUTPUT_CACHE[key] = value
        while len(_FINAL_OUTPUT_CACHE) > FINAL_OUTPUT_CACHE_ITEMS:
            _FINAL_OUTPUT_CACHE.popitem(last=False)


def _processing_interrupted() -> bool:
    """Read ComfyUI's global cancel flag without consuming it."""
    try:
        import comfy.model_management as model_management

        return bool(model_management.processing_interrupted())
    except (ImportError, AttributeError):
        return False


def _throw_if_interrupted() -> None:
    """Raise ComfyUI's native interruption exception so the queue stops cleanly."""
    try:
        import comfy.model_management as model_management

        model_management.throw_exception_if_processing_interrupted()
    except (ImportError, AttributeError):
        return


def _install_llama_interrupt_callback(llm: Any) -> None:
    """Make llama_decode poll ComfyUI's Stop/Interrupt button during GPU or CPU work."""
    try:
        from llama_cpp import llama_cpp as llama_cpp_lib

        callback = llama_cpp_lib.ggml_abort_callback(lambda _data: _processing_interrupted())
        raw_context = getattr(llm.ctx, "ctx", llm.ctx)
        llama_cpp_lib.llama_set_abort_callback(raw_context, callback, None)
        # ctypes callbacks must remain strongly referenced for the lifetime of the C context.
        llm._aimanzi_interrupt_callback = callback
    except (ImportError, AttributeError, TypeError) as exc:
        print(f"[AI蛮子] 警告：当前 llama-cpp-python 无法安装中断回调：{exc}")


def _server_alive(api_base: str, attempts: int = 1, retry_delay: float = 0.2) -> bool:
    """Tolerate transient Windows socket resets during engine startup/reload."""
    for attempt in range(max(1, attempts)):
        _throw_if_interrupted()
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


@functools.lru_cache(maxsize=1)
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


def _effective_ninfer_context(vision: bool, context_tokens: int) -> int:
    """Return the actual capacity passed to the selected bundled engine."""
    profile = _bundled_ninfer_profile()
    limit = profile["vision_context"] if vision else profile["max_context"]
    return min(max(AUTO_CONTEXT_MIN, int(context_tokens)), limit)


@functools.lru_cache(maxsize=1)
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


def _ensure_ninfer_server(model: Path, vision: bool, thinking: bool = True, context_tokens: int = AUTO_CONTEXT_MIN) -> None:
    """Start only a server created by this plugin. Never terminate an unknown user service."""
    kind = "ninfer"
    settings = _settings()
    api_base = settings["ninfer_api_base"]
    # Keep the recorded capacity identical to the command line.  In particular,
    # visual NInfer has a smaller hardware profile cap than text-only NInfer.
    context_tokens = _effective_ninfer_context(vision, context_tokens)
    spec = (str(model), vision, thinking, "", context_tokens)
    old = _OWNED_SERVERS.get(kind)
    old_running = bool(old and _process_is_running(old))
    # Enumerating every TCP connection through psutil is useful only after a
    # ComfyUI restart, when this process no longer has its Popen handle.
    orphan = _orphaned_bundled_ninfer(api_base) if not old_running else None

    # A process can own/listen on the port while briefly resetting health-check
    # connections.  Inspect the process before deciding to launch another copy.
    # This prevents a transient WinError 10054 from turning into a port conflict.
    existing = old if old_running else orphan
    # A cold port normally refuses immediately. Avoid two unnecessary 200ms retry
    # sleeps on the first launch; retain retries only when a real process exists.
    healthy = _server_alive(api_base, attempts=3 if existing is not None else 1)
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
                print(f"[AI蛮子] {kind} 引擎已就绪，复用常驻模型，跳过重新加载。")
                return
            # Allow a busy/reloading service to recover without starting or
            # killing another engine.  Repeated connection resets become a
            # controlled diagnostic instead of leaking a raw socket exception.
            recovery_deadline = time.time() + 15
            while time.time() < recovery_deadline and _process_is_running(existing):
                _throw_if_interrupted()
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
            _SERVER_MODEL_IDS.pop(api_base, None)
    elif healthy:
        # A healthy service not owned by this plugin is intentionally reused.
        return
    command = _bundled_ninfer_command(model, vision, thinking, context_tokens)
    if old and old.poll() is None:
        old.terminate()
    # Only pay the ComfyUI unload/cache-collection cost when a new 27B process
    # will actually be launched. A compatible resident engine returns above.
    _release_comfy_vram_for_ninfer()
    with _FINAL_OUTPUT_CACHE_LOCK:
        _FINAL_OUTPUT_CACHE.clear()
    _SERVER_MODEL_IDS.pop(api_base, None)
    log_path = PLUGIN_DIR / "engine" / f"{kind}-startup.log"
    started_at = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} start ---\n")
        _OWNED_SERVERS[kind] = subprocess.Popen(
            command,
            shell=False,
            cwd=str(PLUGIN_DIR),
            env=_bundled_ninfer_environment(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    _OWNED_SERVER_SPECS[kind] = spec
    deadline = time.time() + 120
    while time.time() < deadline:
        _throw_if_interrupted()
        if _server_alive(api_base):
            print(f"[AI蛮子] {kind} 引擎启动完成，用时 {time.perf_counter() - started_at:.2f} 秒。")
            return
        # Do not make the workflow wait a full two minutes for a deterministic
        # load failure such as insufficient VRAM or a missing DLL.
        if _OWNED_SERVERS[kind].poll() is not None:
            detail = _startup_log_tail(kind)
            _OWNED_SERVERS.pop(kind, None)
            _OWNED_SERVER_SPECS.pop(kind, None)
            raise _ninfer_startup_error(detail)
        elapsed = time.perf_counter() - started_at
        time.sleep(0.1 if elapsed < 5 else 0.25)
    detail = _startup_log_tail(kind)
    raise _ninfer_startup_error(detail, timed_out=True)


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
        _SERVER_MODEL_IDS.clear()
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


def _cuda_runtime_directories() -> list[tuple[Path, str]]:
    """Return trusted CUDA runtime locations in priority order.

    The plugin-owned directory allows release packages to carry NVIDIA's
    redistributable CUDA runtime without modifying ComfyUI or Windows.  Normal
    ComfyUI installations already contain the same CUDA libraries beside
    PyTorch, so reuse them before looking at an optional system CUDA toolkit.
    The NVIDIA driver-only hybrid runtime is discovered in DriverStore because
    it is part of the installed display driver and must not be downloaded from
    an arbitrary DLL website.
    """
    candidates: list[tuple[Path, str]] = [
        (PLUGIN_DIR / "engine" / "llama-cuda-runtime", "插件内置运行库"),
    ]
    for base in (Path(sys.prefix), Path(sys.executable).resolve().parent):
        candidates.extend(
            [
                (
                base / "Lib" / "site-packages" / "nvidia" / "cu13" / "bin" / "x86_64",
                "ComfyUI CUDA 13 运行库",
                ),
                (
                base / "Lib" / "site-packages" / "nvidia" / "cublas" / "bin",
                "ComfyUI NVIDIA Python 运行库",
                ),
            ]
        )
    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        candidates.extend(
            [
                (Path(cuda_path) / "bin" / "x64", "系统 CUDA Toolkit"),
                (Path(cuda_path) / "bin", "系统 CUDA Toolkit"),
            ]
        )
    toolkit_root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if toolkit_root.is_dir():
        for version in sorted(toolkit_root.glob("v*"), reverse=True):
            candidates.extend(
                [
                    (version / "bin" / "x64", f"CUDA Toolkit {version.name}"),
                    (version / "bin", f"CUDA Toolkit {version.name}"),
                ]
            )
    driver_store = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "DriverStore" / "FileRepository"
    if driver_store.is_dir():
        for dll in driver_store.glob("nv*_amd64_*/*nvcudart_hybrid64.dll"):
            candidates.append((dll.parent, "NVIDIA 驱动运行库"))

    result: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, label in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if resolved.is_dir() and key not in seen:
            seen.add(key)
            result.append((resolved, label))
    return result


def _prepare_llama_backend() -> dict[str, Any]:
    """Load llama.cpp's dynamic CUDA backend before the first model is built."""
    global _LLAMA_BACKEND_STATE
    if _LLAMA_BACKEND_STATE is not None:
        return _LLAMA_BACKEND_STATE

    state: dict[str, Any] = {
        "gpu": False,
        "device": "CPU",
        "runtime_sources": [],
        "reason": "CUDA 后端不可用",
    }
    runtime_dirs = _cuda_runtime_directories() if os.name == "nt" else []
    for runtime_dir, label in runtime_dirs:
        try:
            if os.name == "nt" and hasattr(os, "add_dll_directory"):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(runtime_dir)))
            os.environ["PATH"] = str(runtime_dir) + os.pathsep + os.environ.get("PATH", "")
            if any(runtime_dir.glob("cublas64_*.dll")) or any(runtime_dir.glob("nvcudart_hybrid64.dll")):
                state["runtime_sources"].append(label)
        except OSError:
            continue

    try:
        import llama_cpp

        llama_cpp.llama_cpp.llama_backend_init()
        lib_dir = Path(llama_cpp.llama_cpp.__file__).resolve().parent / "lib"
        try:
            from llama_cpp._ggml import ggml_backend_load_all_from_path

            ggml_backend_load_all_from_path(ctypes.c_char_p(str(lib_dir).encode("utf-8")))
        except ImportError:
            # Older statically-linked CUDA wheels do not expose the dynamic loader;
            # llama_supports_gpu_offload() below still reports their GPU capability.
            pass
        state["gpu"] = bool(llama_cpp.llama_cpp.llama_supports_gpu_offload())
        if state["gpu"]:
            state["reason"] = ""
            if torch is not None and torch.cuda.is_available():
                state["device"] = torch.cuda.get_device_name(0)
            else:
                state["device"] = "CUDA GPU"
        else:
            cuda_dll = lib_dir / "ggml-cuda.dll"
            state["reason"] = (
                "llama.cpp CUDA 动态后端未注册"
                if cuda_dll.is_file()
                else "当前 llama-cpp-python 未包含 ggml-cuda.dll"
            )
    except Exception as exc:
        state["reason"] = f"CUDA 后端加载失败：{exc}"

    sources = "、".join(dict.fromkeys(state["runtime_sources"])) or "无"
    if state["gpu"]:
        print(f"[AI蛮子] GGUF 后端：CUDA / {state['device']}；运行库来源：{sources}")
    else:
        print(f"[AI蛮子] GGUF 后端：CPU；原因：{state['reason']}；已检查运行库：{sources}")
    _LLAMA_BACKEND_STATE = state
    return state


def _gguf_block_count(model_path: Path) -> int:
    """Read an early GGUF block-count field without mapping the multi-GB tensor table."""
    try:
        with model_path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                return 0
            version = struct.unpack("<I", stream.read(4))[0]
            if version not in (2, 3):
                return 0
            stream.read(8)  # tensor count
            metadata_count = struct.unpack("<Q", stream.read(8))[0]

            def read_string() -> str:
                length = struct.unpack("<Q", stream.read(8))[0]
                return stream.read(length).decode("utf-8", errors="replace")

            scalar_formats = {
                0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
                6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
            }

            def skip_value(value_type: int) -> None:
                if value_type in scalar_formats:
                    stream.seek(struct.calcsize(scalar_formats[value_type]), 1)
                elif value_type == 8:
                    length = struct.unpack("<Q", stream.read(8))[0]
                    stream.seek(length, 1)
                elif value_type == 9:
                    element_type = struct.unpack("<I", stream.read(4))[0]
                    count = struct.unpack("<Q", stream.read(8))[0]
                    if element_type in scalar_formats:
                        stream.seek(struct.calcsize(scalar_formats[element_type]) * count, 1)
                    else:
                        for _ in range(count):
                            skip_value(element_type)
                else:
                    raise ValueError(f"unsupported GGUF value type: {value_type}")

            for _ in range(min(metadata_count, 64)):
                key = read_string()
                value_type = struct.unpack("<I", stream.read(4))[0]
                if value_type in scalar_formats:
                    fmt = scalar_formats[value_type]
                    value = struct.unpack(fmt, stream.read(struct.calcsize(fmt)))[0]
                    if key.endswith(".block_count"):
                        return int(value)
                elif value_type == 8:
                    read_string()
                else:
                    skip_value(value_type)
    except (OSError, EOFError, struct.error, ValueError):
        pass
    return 0


def _gpu_layer_plan(model_path: Path, mmproj: Path | None, context_tokens: int) -> tuple[list[int], str]:
    """Plan full GPU first, then progressively mixed GPU/CPU fallbacks."""
    backend = _prepare_llama_backend()
    if not backend["gpu"]:
        return [0], str(backend["reason"])

    blocks = _gguf_block_count(model_path)
    free_bytes = 0
    if torch is not None and torch.cuda.is_available():
        try:
            free_bytes = int(torch.cuda.mem_get_info()[0])
        except Exception:
            pass
    # Qwen3-VL 4B uses about 144 KiB of f16 KV per token.  Reserve this plus
    # the projector and 768 MiB for CUDA workspaces/ComfyUI interoperability.
    kv_reserve = min(context_tokens, 32768) * 160 * 1024
    projector_reserve = int(mmproj.stat().st_size * 1.15) if mmproj and mmproj.is_file() else 0
    safety_reserve = 768 * 1024 * 1024
    usable = max(0, free_bytes - kv_reserve - projector_reserve - safety_reserve)
    model_bytes = max(1, model_path.stat().st_size)
    fraction = min(1.0, usable / model_bytes) if free_bytes else 1.0

    if blocks:
        estimated = max(1, min(blocks, int(blocks * fraction)))
        plan = [-1] if estimated >= blocks else [estimated]
        plan.extend(
            candidate
            for candidate in (int(blocks * ratio) for ratio in (0.75, 0.5, 0.25))
            if candidate < estimated
        )
    else:
        plan = [-1, 32, 24, 16, 8]
    plan.append(0)
    unique: list[int] = []
    for item in plan:
        if item not in unique and (item == -1 or item >= 0):
            unique.append(item)
    free_gib = free_bytes / (1024 ** 3) if free_bytes else 0.0
    return unique, f"可用显存 {free_gib:.2f} GiB，计划 GPU 层 {unique}"


def _gpu_retryable_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "cuda", "out of memory", "allocation", "alloc", "buffer", "vram",
            "not enough memory", "显存", "内存不足",
        )
    )


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
        _throw_if_interrupted()
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
    api_base: str,
    model: str,
    instruction: str,
    image_urls: list[str],
    allow_reasoning: bool = False,
    api_key: str = "",
    observation_mode: bool = False,
    inference_strategy: str = "普通推理",
    seed: int = 0,
) -> str:
    seed = int(seed) & SEED_MAX
    temperature, top_p, strategy_instruction = _inference_profile(inference_strategy, observation_mode)
    strategy_instruction += _creative_seed_instruction(inference_strategy, observation_mode, seed)
    parsed_base = urlsplit(api_base)
    local_service = parsed_base.hostname in {"127.0.0.1", "localhost", "::1"}
    response_cache_key = None
    if local_service:
        response_cache_key = _visual_cache_key(
            "local-api-output-v1",
            image_urls,
            api_base.rstrip("/"),
            model,
            instruction,
            allow_reasoning,
            observation_mode,
            inference_strategy,
            seed,
        )
        cached_answer = _final_cache_get(response_cache_key)
        if cached_answer is not None:
            print("[AI蛮子] 完全相同的本地 API 请求缓存命中，跳过重复生成。")
            return cached_answer
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
                        (
                            "你是视频视觉观察助手。只按输入帧的先后顺序记录实际可见事实，"
                            "特别关注连续动作、镜头运动、场景变化和首尾状态；不要生成最终提示词。"
                            if observation_mode else
                            "你是提示词助手。默认只输出可直接使用的正向提示词纯文本，不添加解释、标题或 Markdown。"
                            "可以在内部思考，但最终消息严禁包含分析、推理步骤、思考过程或总结；"
                            "只有用户明确要求其他输出格式或展示分析时，才遵循用户的额外约束。"
                        )
                        + (
                            "本轮已附带图像或视频画面，必须以媒体中实际可见的像素内容为首要事实，"
                            "不得把模板中的示例描述当成画面内容，也绝不能要求用户再次上传媒体。"
                            if resized_urls else ""
                        )
                        + " " + strategy_instruction
                    ),
                },
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "stream": False,
        }
        try:
            endpoint = api_base.strip().rstrip("/")
            if not endpoint.lower().endswith("/chat/completions"):
                endpoint += "/chat/completions"
            auth_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
            result = _json_request(
                endpoint,
                payload,
                int(_settings()["request_timeout_seconds"]),
                auth_headers,
            )
            break
        except RuntimeError as exc:
            seed_error = str(exc).lower()
            if "seed" in seed_error and any(
                marker in seed_error
                for marker in (
                    "unsupported", "unknown", "unrecognized", "not allowed", "not permitted",
                    "unexpected", "extra_forbidden", "不支持", "未知",
                )
            ):
                # Some OpenAI-compatible providers reject the otherwise standard
                # seed field. Retry once without it instead of failing the workflow.
                payload.pop("seed", None)
                print("[AI蛮子] 当前在线服务不支持 seed 参数，本次已自动无种子重试。")
                try:
                    result = _json_request(
                        endpoint,
                        payload,
                        int(_settings()["request_timeout_seconds"]),
                        auth_headers,
                    )
                    break
                except RuntimeError as retry_exc:
                    exc = retry_exc
            last_error = exc
            if "media_budget_exceeded" not in str(exc) or attempt == len(levels) - 1:
                raise
    if result is None:
        raise last_error or RuntimeError("推理服务没有返回结果。")
    try:
        choice = result["choices"][0]
        message = choice["message"]
        raw_content = message.get("content", "") if isinstance(message, dict) else ""
        answer = _clean_answer(raw_content, allow_reasoning)
        if not answer:
            reasoning = ""
            if isinstance(message, dict):
                reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "").strip()
            finish_reason = str(choice.get("finish_reason", "")) if isinstance(choice, dict) else ""
            finish_lower = finish_reason.lower()
            if reasoning or "context" in finish_lower or finish_lower in {"length", "max_tokens"}:
                raise RuntimeError(
                    "模型的思考过程已占满当前上下文，尚未生成最终答案。[empty_final_after_reasoning]"
                )
        if answer and response_cache_key:
            _final_cache_put(response_cache_key, answer)
        return answer
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


def _video_batch(images: Any) -> Any:
    """Validate a decoded ComfyUI video-frame IMAGE batch without copying it."""
    if torch is not None and isinstance(images, torch.Tensor):
        array = images
    else:
        array = np.asarray(images) if images is not None else None
    if array is None or array.ndim != 4 or array.shape[0] < 1 or array.shape[-1] not in (3, 4):
        raise ValueError("视频必须连接加载视频节点输出的 IMAGE 批次，格式应为 [帧数, 高, 宽, RGB/RGBA]。")
    return array


def _video_frame_signature(frame: Any) -> np.ndarray:
    """Create a tiny luminance signature used only for scene-change ranking."""
    if torch is not None and isinstance(frame, torch.Tensor):
        height, width = int(frame.shape[0]), int(frame.shape[1])
        sample = frame[::max(1, height // 32), ::max(1, width // 32), :3].detach().float().cpu().numpy()
    else:
        source = np.asarray(frame)
        height, width = source.shape[:2]
        sample = source[::max(1, height // 32), ::max(1, width // 32), :3].astype(np.float32, copy=False)
    if sample.shape[0] > 32 or sample.shape[1] > 32:
        sample = sample[:32, :32]
    return sample.mean(axis=2)


def _select_video_frame_indexes(images: Any, precision: str) -> list[int]:
    """Blend timeline coverage with visual-change frames while preserving order."""
    batch = _video_batch(images)
    count = int(batch.shape[0])
    limit = min(count, VIDEO_ANALYSIS_LIMITS.get(precision, 64))
    if count <= limit:
        return list(range(count))

    # Roughly two thirds of the budget guarantees full timeline coverage. The
    # rest is spent on cuts/action changes that uniform sampling can miss.
    uniform_count = max(2, int(round(limit * 0.65)))
    selected = set(np.linspace(0, count - 1, uniform_count, dtype=int).tolist())
    change_slots = max(0, limit - len(selected))
    if change_slots:
        scores: list[tuple[float, int]] = []
        previous = _video_frame_signature(batch[0])
        for index in range(1, count):
            _throw_if_interrupted()
            current = _video_frame_signature(batch[index])
            common_h = min(previous.shape[0], current.shape[0])
            common_w = min(previous.shape[1], current.shape[1])
            score = float(np.mean(np.abs(current[:common_h, :common_w] - previous[:common_h, :common_w])))
            scores.append((score, index))
            previous = current
        for _, index in sorted(scores, reverse=True):
            selected.add(index)
            if len(selected) >= limit:
                break
    selected.update((0, count - 1))
    return sorted(selected)[:limit]


def _video_frame_urls(images: Any, precision: str, max_pixels: int = 768 * 768) -> tuple[list[str], list[int], int]:
    batch = _video_batch(images)
    indexes = _select_video_frame_indexes(batch, precision)
    urls: list[str] = []
    for index in indexes:
        _throw_if_interrupted()
        urls.append(_tensor_data_url(batch[index], max_pixels))
    return urls, indexes, int(batch.shape[0])


def _video_segment_instruction(
    user_instruction: str, frame_indexes: list[int], total_frames: int, segment_number: int, segment_count: int,
) -> str:
    labels = "、".join(str(index + 1) for index in frame_indexes)
    return (
        f"这是按原始时间顺序排列的视频第 {segment_number}/{segment_count} 段，"
        f"本段包含原视频第 {labels} 帧，原视频共 {total_frames} 帧。"
        "请只记录画面中实际可见的事实，并结合相邻帧判断主体动作、物体运动、镜头运动、"
        "景别、场景变化、光线、色彩、风格和可辨识文字。不要生成最终提示词，不要要求重新上传视频，"
        "不要把用户要求中的示例当作画面事实。单帧无法证明的运动必须标记为不确定。\n\n"
        "用户最终任务（仅用于确定观察重点）：\n" + user_instruction
    )


def _video_facts_from_frames(
    frame_urls: list[str], frame_indexes: list[int], total_frames: int, user_instruction: str,
    analyze_segment: Any, base_seed: int = 0, cache_namespace: str | None = None,
) -> str:
    """Analyze chronological chunks, then return compact evidence for final synthesis."""
    if not frame_urls:
        raise ValueError("视频 IMAGE 批次为空。")
    cache_key = None
    if cache_namespace:
        cache_key = _visual_cache_key(
            cache_namespace,
            frame_urls,
            tuple(frame_indexes),
            total_frames,
            user_instruction,
            int(base_seed) & SEED_MAX,
            VIDEO_SEGMENT_FRAMES,
        )
        cached = _visual_cache_get(cache_key)
        if cached is not None:
            print("[AI蛮子] 视频视觉事实缓存命中，跳过重复逐帧识别。")
            return cached
    segment_count = int(np.ceil(len(frame_urls) / VIDEO_SEGMENT_FRAMES))
    facts: list[str] = []
    for offset in range(0, len(frame_urls), VIDEO_SEGMENT_FRAMES):
        _throw_if_interrupted()
        urls = frame_urls[offset:offset + VIDEO_SEGMENT_FRAMES]
        indexes = frame_indexes[offset:offset + VIDEO_SEGMENT_FRAMES]
        number = offset // VIDEO_SEGMENT_FRAMES + 1
        observation = analyze_segment(
            _video_segment_instruction(user_instruction, indexes, total_frames, number, segment_count),
            urls,
            (int(base_seed) + number - 1) & SEED_MAX,
        )
        _throw_if_interrupted()
        if not str(observation).strip():
            raise RuntimeError(f"视频第 {number}/{segment_count} 段没有返回有效识别结果。")
        facts.append(f"[时间段 {number}/{segment_count}，原始帧 {indexes[0] + 1}-{indexes[-1] + 1}]\n{observation.strip()}")
    result = "\n\n".join(facts)
    if cache_key:
        _visual_cache_put(cache_key, result)
    return result


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


def _ninfer_visual_facts(instruction: str, image_urls: list[str], seed: int = 0) -> str:
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
    cache_key = _visual_cache_key(
        "ninfer-image-bridge-v1",
        grounded_media,
        _path_identity(bridge_model),
        _path_identity(bridge_mmproj),
        observation_request,
        # Visual observation is factual and intentionally independent from the
        # final creative seed, so Random/Increment does not invalidate grounding.
        0,
    )
    cached = _visual_cache_get(cache_key)
    if cached is not None:
        print("[AI蛮子] 图像视觉事实缓存命中，跳过重复视觉识别。")
        return cached
    facts = _llamacpp_chat(
        bridge_model,
        observation_request,
        grounded_media,
        bridge_mmproj,
        False,
        observation_mode=True,
        inference_strategy="普通推理",
        seed=0,
    )
    if not facts.strip():
        raise RuntimeError("视觉桥接模型没有返回可用的图像/视频识别结果。")
    _visual_cache_put(cache_key, facts)
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
    observation_mode: bool = False,
    inference_strategy: str = "普通推理",
    seed: int = 0,
) -> str:
    """Universal GGUF backend with automatic CUDA, mixed, and CPU execution."""
    # ComfyUI/Torch can preload Intel OpenMP while the installed llama-cpp-python
    # wheel loads LLVM OpenMP.  Without this compatibility flag Windows aborts with
    # OMP Error #15 before any GGUF request can run.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    backend = _prepare_llama_backend()
    try:
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Qwen3VLChatHandler, Qwen35ChatHandler
    except ImportError as exc:
        raise RuntimeError("关闭 NInfer 的 GGUF 推理需要 ComfyUI Python 中的 llama-cpp-python。") from exc
    seed = int(seed) & SEED_MAX
    temperature, top_p, strategy_instruction = _inference_profile(inference_strategy, observation_mode)
    strategy_instruction += _creative_seed_instruction(inference_strategy, observation_mode, seed)
    context_tokens = context_tokens or _auto_context_tokens(instruction, len(image_urls))
    model_prefix = (str(model_path), str(mmproj or ""), thinking)
    key: tuple[str, str, bool, int, str, int] | None = None
    llm = None
    compatible = [
        (cached_key, cached_model)
        for cached_key, cached_model in _LLAMA_CPP_MODELS.items()
        if cached_key[:3] == model_prefix and cached_key[3] >= context_tokens
    ]
    if compatible:
        # A larger already-loaded context can safely serve a smaller request. Choose
        # the smallest compatible instance to minimize retained KV memory.
        key, llm = min(compatible, key=lambda item: item[0][3])
        if key[3] > context_tokens:
            print(f"[AI蛮子] 复用已加载的 {key[3]} 上下文实例，跳过 {context_tokens} 上下文重载。")
    if llm is None:
        if image_urls and mmproj is None:
            raise RuntimeError("图像或视频 GGUF 推理需要匹配的 mmproj 文件。")
        if backend["gpu"]:
            # Do not leave a previously selected GGUF occupying VRAM while a new
            # model/context is being loaded. Matching requests were returned from
            # the cache above, so everything left here is safe to replace.
            if _LLAMA_CPP_MODELS:
                _unload_plugin_model(None)
            # ComfyUI normally keeps diffusion models resident. Release those allocations
            # before measuring capacity so llama.cpp can put the maximum safe layer count
            # on GPU rather than needlessly falling back to CPU.
            _release_comfy_vram_for_ninfer()
        layer_plan, plan_detail = _gpu_layer_plan(model_path, mmproj if image_urls else None, context_tokens)
        print(f"[AI蛮子] GGUF 显存规划：{plan_detail}")
        attempts: list[tuple[int, bool]] = [(layers, bool(backend["gpu"])) for layers in layer_plan]
        if backend["gpu"] and image_urls:
            # Last-resort compatibility path when even the projector cannot fit on GPU.
            attempts.append((0, False))
        last_error: Exception | None = None
        lowered = model_path.name.lower()
        for layers, vision_gpu in attempts:
            handler = None
            candidate = None
            try:
                if image_urls:
                    if "qwen3.5" in lowered:
                        handler = Qwen35ChatHandler(
                            clip_model_path=str(mmproj),
                            enable_thinking=thinking,
                            use_gpu=vision_gpu,
                            verbose=False,
                            image_min_tokens=1024,
                        )
                    elif "qwen" in lowered and "vl" in lowered:
                        handler = Qwen3VLChatHandler(
                            clip_model_path=str(mmproj),
                            force_reasoning=thinking,
                            use_gpu=vision_gpu,
                            verbose=False,
                            image_min_tokens=1024,
                        )
                    else:
                        raise RuntimeError("当前仅内置 Qwen3-VL 与 Qwen3.5 GGUF 的视觉处理器。")
                candidate = Llama(
                    model_path=str(model_path),
                    chat_handler=handler,
                    n_ctx=context_tokens,
                    n_gpu_layers=layers,
                    n_threads=max(1, (os.cpu_count() or 4) - 1),
                    verbose=False,
                )
                _install_llama_interrupt_callback(candidate)
                llm = candidate
                mode = "全 GPU" if layers == -1 else ("GPU+CPU 混合" if layers > 0 or vision_gpu else "纯 CPU")
                layer_text = "全部" if layers == -1 else str(layers)
                vision_text = "GPU" if image_urls and vision_gpu else ("CPU" if image_urls else "无媒体")
                print(f"[AI蛮子] GGUF 已加载：{mode}；主模型 GPU 层：{layer_text}；视觉编码器：{vision_text}")
                key = model_prefix + (context_tokens, f"{mode}/{vision_text}", layers)
                _LLAMA_CPP_MODELS[key] = llm
                break
            except Exception as exc:
                last_error = exc
                if candidate is not None:
                    try:
                        candidate.close()
                    except Exception:
                        pass
                del handler
                gc.collect()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if (layers != 0 or vision_gpu) and _gpu_retryable_error(exc):
                    print(f"[AI蛮子] GPU 加载未成功，自动降低 GPU 占用后重试：{exc}")
                    continue
                raise RuntimeError(f"GGUF 模型加载失败：{exc}") from exc
        if llm is None:
            raise RuntimeError(f"GGUF 模型加载失败：{last_error}") from last_error
    content: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
    content.extend({"type": "image_url", "image_url": {"url": item}} for item in image_urls)
    final_cache_key = _visual_cache_key(
        "llamacpp-output-v1",
        image_urls,
        _path_identity(model_path),
        _path_identity(mmproj),
        instruction,
        thinking,
        context_tokens,
        allow_reasoning,
        observation_mode,
        inference_strategy,
        seed,
    )
    cached_answer = _final_cache_get(final_cache_key)
    if cached_answer is not None:
        print("[AI蛮子] 完全相同的 GGUF 请求缓存命中，跳过重复生成。")
        return cached_answer
    try:
        _throw_if_interrupted()
        # Keep n_tokens intact: llama.cpp compares the next prompt with its current
        # token history and reuses the longest KV prefix. Exact fixed-seed requests
        # are served by the result cache above, preserving reproducibility without
        # disabling native prefix reuse for changed prompts.
        llm.set_seed(seed)
        result = llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        (
                            "你是视频视觉观察助手。只按输入帧的先后顺序记录实际可见事实，"
                            "特别关注连续动作、镜头运动、场景变化和首尾状态；不要生成最终提示词。"
                            if observation_mode else
                            "你是提示词助手。默认只输出可直接使用的正向提示词纯文本，不添加解释、标题或 Markdown。"
                            "可以在内部思考，但最终消息严禁包含分析、推理步骤、思考过程或总结；"
                            "只有用户明确要求其他输出格式或展示分析时，才遵循用户的额外约束。"
                        )
                        + (
                            "本轮已附带图像或视频画面，必须以媒体中实际可见的像素内容为首要事实，"
                            "不得把模板中的示例描述当成画面内容，也绝不能要求用户再次上传媒体。"
                            if image_urls else ""
                        )
                        + " " + strategy_instruction
                    ),
                },
                {"role": "user", "content": content if image_urls else instruction},
            ],
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=None,
        )
        _throw_if_interrupted()
        answer = _clean_answer(result["choices"][0]["message"]["content"], allow_reasoning)
        if answer:
            _final_cache_put(final_cache_key, answer)
        return answer
    except ValueError as exc:
        _throw_if_interrupted()
        message = str(exc)
        match = re.search(r"Requested tokens \((\d+)\) exceed context window", message)
        if match:
            required_tokens = int(match.group(1))
            next_context = _auto_context_tokens(instruction, len(image_urls), required_tokens)
            if next_context > context_tokens:
                # The exact chat template can be longer than the pre-load estimate. Reload once at
                # the required size rather than forcing users to choose a context manually.
                loaded = _LLAMA_CPP_MODELS.pop(key, None) if key is not None else None
                try:
                    if loaded is not None:
                        loaded.close()
                except Exception:
                    pass
                return _llamacpp_chat(
                    model_path,
                    instruction,
                    image_urls,
                    mmproj,
                    thinking,
                    next_context,
                    allow_reasoning,
                    observation_mode,
                    inference_strategy,
                    seed,
                )
            raise RuntimeError(
                f"GGUF 输入需要至少 {required_tokens} tokens，已达到模型/硬件自动上下文上限 "
                f"{AUTO_CONTEXT_MAX} tokens。请缩短 TXT/MD 模板、减少图片或关闭思考模式后重试。"
            ) from exc
        raise RuntimeError(f"GGUF 推理失败：{exc}") from exc
    except Exception as exc:
        _throw_if_interrupted()
        raise RuntimeError(f"GGUF 推理失败：{exc}") from exc


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


def _template_payload(core: str, references: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "core_text": core,
        "references": references,
    }


def _flatten_template_payload(payload: dict[str, Any]) -> str:
    """Build the one and only text shown by the loader and sent to the model."""
    text = str(payload.get("core_text", "")).strip()
    references = payload.get("references", [])
    if isinstance(references, list):
        for item in references:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip() or "未命名资料"
            content = str(item.get("text", "")).strip()
            if content:
                text += f"\n\n[Skill 参考资料：{name}]\n{content}"
    return text.strip()


def _read_skill_archive(path: Path) -> dict[str, Any]:
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
        references: list[dict[str, str]] = []
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
                raise ValueError("Skill 文本资料总量超过 8 MiB 安全上限。")
            content = _decode_template_text(data, member_name)
            references.append({"name": relative.as_posix(), "text": content})
            text_bytes += len(data)

    return _template_payload(skill_text, references)


def _read_template_or_skill_uncached(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        if path.stat().st_size > SKILL_MAX_TEXT_BYTES:
            raise ValueError("模板文件超过 8 MiB 安全上限。")
        text = _decode_template_text(path.read_bytes(), path.name)
        return _template_payload(text, [])
    if suffix == ".skill" and not zipfile.is_zipfile(path):
        if path.stat().st_size > SKILL_MAX_TEXT_BYTES:
            raise ValueError("Skill 文本文件超过 8 MiB 安全上限。")
        text = _decode_template_text(path.read_bytes(), path.name)
        return _template_payload(text, [])
    if suffix in {".skill", ".zip"}:
        return _read_skill_archive(path)
    raise ValueError("仅支持 txt、md、markdown、skill 或包含 SKILL.md 的 zip 文件。")


@functools.lru_cache(maxsize=32)
def _read_template_cached(path_text: str, modified_ns: int, size: int) -> str:
    # modified_ns and size intentionally participate in the cache key so replacing an
    # uploaded file with the same name cannot return stale content.
    del modified_ns, size
    return _flatten_template_payload(_read_template_or_skill_uncached(Path(path_text)))


def _read_template_or_skill(path: Path) -> str:
    stat = path.stat()
    return _read_template_cached(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


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
                "推理策略": (list(INFERENCE_STRATEGIES), {"default": "普通推理"}),
                "随机种子": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": SEED_MAX,
                    "control_after_generate": True,
                }),
                "推理方式": (["本地推理", "在线推理"], {"default": "本地推理"}),
                "在线_API_URL": ("STRING", {"default": "https://api.openai.com/v1"}),
                "在线_API_Key": ("STRING", {"default": "", "password": True}),
                "在线_模型_ID": ("STRING", {"default": ""}),
                "启用_ninfer": ("BOOLEAN", {"default": True}),
                "启用思考模式": ("BOOLEAN", {"default": True}),
                "推理后卸载模型": ("BOOLEAN", {"default": False}),
                "视频分析精度": (list(VIDEO_ANALYSIS_LIMITS), {"default": "标准（最多64帧）"}),
                "模型": (_model_choices(),),
                "mmproj": (_mmproj_choices(),),
            },
            # The frontend reveals the next image socket after the preceding
            # one is connected.  This special mapping makes every revealed
            # 图像_1 ... 图像_10 socket a real, validated ComfyUI IMAGE input.
            "optional": _DynamicImageOptionalInputs({
                "模板输入": ("AIMANZI_TEMPLATE",),
                "图像_1": ("IMAGE",),
                "视频": ("IMAGE",),
            }),
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    def generate(self, **kwargs):
        request_started = time.perf_counter()
        _throw_if_interrupted()
        inference_mode = str(kwargs.get("推理方式", "本地推理"))
        use_online = inference_mode == "在线推理"
        use_ninfer = bool(kwargs["启用_ninfer"])
        thinking = bool(kwargs.get("启用思考模式", True))
        unload_after = bool(kwargs.get("推理后卸载模型", False))
        inference_strategy = str(kwargs.get("推理策略", "普通推理"))
        if inference_strategy not in INFERENCE_STRATEGIES:
            inference_strategy = "普通推理"
        seed = int(kwargs.get("随机种子", 0)) & SEED_MAX
        print(f"[AI蛮子] 后端收到本轮种子：{seed}（{inference_strategy}）")
        user_instruction = str(kwargs.get("文字要求", "")).strip()
        instruction = user_instruction
        # Decide this from the user's own text before appending templates or
        # machine-generated visual facts, so a Skill cannot expose chain-of-thought.
        allow_reasoning = _user_requests_reasoning(user_instruction)
        template_value = kwargs.get("模板输入")
        template_text = str(template_value or "").strip()
        if template_text:
            # The loader's visible parsed text is the sole template value. Do not
            # reopen, select, summarize or otherwise rewrite it in this node.
            instruction = (
                "以下是用户提供的模板/Skill 原文规则。完整遵守这些规则；其中的示例不得覆盖"
                "本次媒体事实或用户的明确要求：\n"
                + template_text
                + "\n\n[用户本次要求]\n"
                + user_instruction
            )
        template_ready = time.perf_counter()
        image_urls: list[str] = []
        # NInfer media is grounded by the Qwen bridge at about 1MP per item.
        # Apply that cap while converting the Comfy tensor, not after creating
        # a full-resolution PNG/data URL. GGUF also gets a conservative 2MP
        # pre-cap before the later shared 6MP budget is enforced.
        input_item_pixels = 1024 * 1024 if (use_ninfer or use_online) else 2 * 1024 * 1024
        for key in sorted(
            (item for item in kwargs if re.fullmatch(r"图像_\d+", item) and int(item.split("_")[-1]) <= MAX_DYNAMIC_IMAGE_INPUTS),
            key=lambda item: int(item.split("_")[-1]),
        ):
            _throw_if_interrupted()
            value = kwargs.get(key)
            if value is None:
                continue
            # Comfy IMAGE may contain a batch; retain all batch images in socket order.
            if (torch is not None and isinstance(value, torch.Tensor) and value.ndim == 4) or (
                isinstance(value, np.ndarray) and value.ndim == 4
            ):
                for index in range(value.shape[0]):
                    _throw_if_interrupted()
                    image_urls.append(_tensor_data_url(value[index], input_item_pixels))
            else:
                image_urls.append(_tensor_data_url(value, input_item_pixels))
        video_urls: list[str] = []
        video_indexes: list[int] = []
        video_total_frames = 0
        video_frames = kwargs.get("视频")
        if video_frames is not None:
            video_urls, video_indexes, video_total_frames = _video_frame_urls(
                video_frames,
                str(kwargs.get("视频分析精度", "标准（最多64帧）")),
            )
        media_ready = time.perf_counter()
        backend_kind: str | None = None
        try:
            if use_online:
                api_base = str(kwargs.get("在线_API_URL", "")).strip()
                api_key = str(kwargs.get("在线_API_Key", "")).strip()
                model_id = str(kwargs.get("在线_模型_ID", "")).strip()
                parsed_api = urlsplit(api_base)
                if parsed_api.scheme not in {"http", "https"} or not parsed_api.netloc:
                    raise ValueError("在线推理的 API URL 无效，请填写 http:// 或 https:// 开头的地址。")
                if not model_id:
                    raise ValueError("在线推理必须填写模型 ID。")
                if video_urls:
                    video_facts = _video_facts_from_frames(
                        video_urls,
                        video_indexes,
                        video_total_frames,
                        user_instruction,
                        lambda prompt, media, segment_seed: _api_chat(
                            api_base,
                            model_id,
                            prompt,
                            media,
                            False,
                            api_key,
                            True,
                            "普通推理",
                            segment_seed,
                        ),
                        0,
                    )
                    instruction += (
                        "\n\n以下是视觉模型按原始时间顺序对视频各段的观察记录。"
                        "请综合全部时间段，重点还原动作发展、镜头运动、转场和首尾变化；"
                        "默认只输出最终视频提示词：\n" + video_facts
                    )
                image_urls = _resize_media_to_budget(
                    image_urls,
                    VISION_SAFE_TOTAL_PIXELS,
                    1024 * 1024,
                )
                answer = _api_chat(
                    api_base,
                    model_id,
                    instruction,
                    image_urls,
                    allow_reasoning,
                    api_key,
                    False,
                    inference_strategy,
                    seed,
                )
            else:
                model, mmproj = _validate_model(
                    str(kwargs["模型"]), use_ninfer, bool(image_urls or video_urls), str(kwargs["mmproj"])
                )
                if use_ninfer:
                    backend_kind = "ninfer"
                    # Direct vision in some ternary NInfer artifacts is structurally present but
                    # semantically misaligned with the converted language body. Ground the media
                    # with a local Qwen vision GGUF, then let NInfer do the final writing pass.
                    if video_urls:
                        bridge_model, bridge_mmproj = _find_visual_bridge_model()
                        video_facts = _video_facts_from_frames(
                            video_urls,
                            video_indexes,
                            video_total_frames,
                            user_instruction,
                            lambda prompt, media, segment_seed: _llamacpp_chat(
                                bridge_model,
                                prompt,
                                _resize_media_to_budget(media, 4 * 1024 * 1024, 768 * 768),
                                bridge_mmproj,
                                False,
                                observation_mode=True,
                                inference_strategy="普通推理",
                                seed=segment_seed,
                            ),
                            0,
                            cache_namespace=(
                                "ninfer-video-bridge-v1|"
                                + _path_identity(bridge_model)
                                + "|"
                                + _path_identity(bridge_mmproj)
                            ),
                        )
                        instruction += (
                            "\n\n以下内容由视觉桥接模型按时间顺序分析本次视频帧得到，"
                            "是必须遵守的视频事实。请综合动作发展、镜头运动、场景变化和首尾状态，"
                            "默认只输出最终视频提示词：\n" + video_facts
                        )
                    if image_urls:
                        visual_facts = _ninfer_visual_facts(user_instruction, image_urls, seed)
                        instruction += (
                            "\n\n以下内容由视觉桥接模型从本次输入图像/视频像素中提取，"
                            "是生成结果时必须遵守的画面事实。不得声称未收到媒体，也不得用模板示例覆盖这些事实：\n"
                            + visual_facts
                        )
                        image_urls = []
                    # Close any cached llama.cpp model before NInfer reserves
                    # host/device memory. Media bridges have completed by here.
                    _unload_plugin_model(None)
                    context_tokens = _ninfer_context_tokens(instruction, thinking)
                    _ensure_ninfer_server(model, False, thinking=thinking, context_tokens=context_tokens)
                    api_base = _settings()["ninfer_api_base"]
                    try:
                        answer = _api_chat(
                            api_base,
                            _server_model_id(api_base, model.name),
                            instruction,
                            [],
                            allow_reasoning,
                            inference_strategy=inference_strategy,
                            seed=seed,
                        )
                    except RuntimeError as exc:
                        if "[empty_final_after_reasoning]" not in str(exc) or not thinking:
                            raise
                        retry_context = min(AUTO_CONTEXT_MAX, max(32768, context_tokens * 2))
                        if retry_context <= context_tokens:
                            raise RuntimeError(
                                "NInfer 已达到最大上下文，但思考过程仍未生成最终答案。"
                                "请关闭思考模式或缩短模板。"
                            ) from exc
                        print(
                            f"[AI蛮子] NInfer 思考耗尽 {context_tokens} 上下文，"
                            f"自动扩容到 {retry_context} 并重试一次。"
                        )
                        _ensure_ninfer_server(model, False, thinking=thinking, context_tokens=retry_context)
                        answer = _api_chat(
                            api_base,
                            _server_model_id(api_base, model.name),
                            instruction,
                            [],
                            allow_reasoning,
                            inference_strategy=inference_strategy,
                            seed=seed,
                        )
                else:
                    backend_kind = "llama_cpp"
                    if video_urls:
                        video_facts = _video_facts_from_frames(
                            video_urls,
                            video_indexes,
                            video_total_frames,
                            user_instruction,
                            lambda prompt, media, segment_seed: _llamacpp_chat(
                                model,
                                prompt,
                                _resize_media_to_budget(media, 4 * 1024 * 1024, 768 * 768),
                                mmproj,
                                thinking,
                                observation_mode=True,
                                inference_strategy="普通推理",
                                seed=segment_seed,
                            ),
                            0,
                            cache_namespace=(
                                "gguf-video-v1|"
                                + _path_identity(model)
                                + "|"
                                + _path_identity(mmproj)
                                + f"|thinking={thinking}"
                            ),
                        )
                        instruction += (
                            "\n\n以下是视觉模型按时间顺序对视频各段的观察记录。"
                            "请综合全部时间段，重点还原动作发展、镜头运动、转场和首尾变化；"
                            "默认只输出最终视频提示词：\n" + video_facts
                        )
                    image_urls = _resize_media_to_budget(image_urls, VISION_SAFE_TOTAL_PIXELS)
                    answer = _llamacpp_chat(
                        model,
                        instruction,
                        image_urls,
                        mmproj,
                        thinking,
                        allow_reasoning=allow_reasoning,
                        inference_strategy=inference_strategy,
                        seed=seed,
                    )
        finally:
            if unload_after:
                _unload_plugin_model(backend_kind)
        if not answer:
            raise RuntimeError("模型没有返回正向提示词。")
        finished = time.perf_counter()
        print(
            f"[AI蛮子] 本轮耗时：模板解析 {template_ready - request_started:.3f}s，"
            f"媒体预处理 {media_ready - template_ready:.3f}s，"
            f"引擎推理 {finished - media_ready:.3f}s，总计 {finished - request_started:.3f}s。"
        )
        return (answer,)


def _resolve_uploaded_template(selected_file: str) -> Path:
    if not selected_file.strip() or selected_file == "点击上传 TXT/MD 文件" or folder_paths is None:
        raise ValueError("请使用节点上的上传按钮选择模板或 Skill 文件。")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = (root / selected_file).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("模板/Skill 文件无效或不在 ComfyUI/input 目录中。")
    return path


async def _parse_template_route(request: Any) -> Any:
    try:
        body = await request.json()
        selected_file = str(body.get("file", ""))
        path = _resolve_uploaded_template(selected_file)
        text = _read_template_or_skill(path)
        if not text.strip():
            raise ValueError("模板/Skill 解析后没有可提交的文字内容。")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(
            f"[AI蛮子] 模板已解析：{path.name}；最终 {len(text)} 字符；"
            f"SHA256 {digest[:12]}。"
        )
        return web.json_response({
            "ok": True,
            "file": selected_file,
            "content": text,
            "characters": len(text),
            "sha256": digest,
        })
    except (ValueError, OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        return web.json_response({"ok": False, "error": str(exc)}, status=400)


if PromptServer is not None and web is not None and getattr(PromptServer, "instance", None) is not None:
    PromptServer.instance.routes.post("/aimanzi/parse-template")(_parse_template_route)


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
        return {"required": {
            "TXT/MD 文件": ("STRING", {"default": ""}),
            "提交给模型的内容": ("STRING", {"multiline": True, "default": ""}),
        }}

    def read(self, **kwargs):
        # Accept the former widget key so saved workflows continue to execute.
        if "提交给模型的内容" in kwargs:
            content = str(kwargs.get("提交给模型的内容", "")).strip()
            if not content:
                raise ValueError("解析内容为空。请重新上传模板/Skill，或在内容框中输入文字。")
            return (content,)
        # Compatibility fallback for workflows saved before immediate parsing.
        selected_file = str(kwargs.get("模板/Skill 文件", kwargs.get("TXT/MD 文件", "")))
        path = _resolve_uploaded_template(selected_file)
        content = _read_template_or_skill(path)
        if not content.strip():
            raise ValueError("模板/Skill 解析后没有可提交的文字内容。")
        return (content,)


NODE_CLASS_MAPPINGS = {
    "AIManziMultimodalPrompt": AIManziMultimodalPrompt,
    "AIManziReadText": AIManziReadText,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AIManziMultimodalPrompt": "AI蛮子 多模态提示词工作台",
    "AIManziReadText": "AI蛮子 加载模板 / Skill",
}
