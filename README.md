# AI蛮子多模态提示词插件

放入 `ComfyUI/custom_nodes` 后重启 ComfyUI。模型默认扫描 `ComfyUI/models/LLM`。

## 插件及模型下载

**AI蛮子 多模态工作台插件及模型：** [夸克网盘下载](https://pan.quark.cn/s/be99d9acf669)

## 使用

- 工作台顶部是 **文字要求**；模型与 NInfer 参数在它下方。输出仅为正向提示词纯文本。
- **模板输入** 是唯一的模板连线端口。使用 `AI蛮子 加载模板 / Skill` 节点上传 txt、md、markdown、skill，或包含 `SKILL.md` 的 zip 文件；上传后输出直接接到该端口。
- `.skill` 按 ZIP 格式安全解析：完整读取唯一的 `SKILL.md`，并合并同一 Skill 下 `references/` 中的 md/txt/json/yaml 文本。`scripts/` 与 `assets/` 不会执行或注入模型；路径穿越、符号链接、加密包、异常压缩比、过多文件和超大内容会被拒绝。
- **视频输入** 是标准 `VIDEO` 连线端口，可连接 ComfyUI 或 ComfyUI-Easy-Media 的 VIDEO 输出；不需要填写视频路径。
- 图像接口按连接顺序自动展开，最多支持 **10 个 IMAGE 接口**；一个 IMAGE 批次也会逐张发送。视频自动均匀抽取最多 10 张代表帧，覆盖整段视频。
- 图像与视频帧会在推理前保持宽高比自动缩放。原始图像和视频文件不会被修改。
- 开启 **启用 NInfer**：选择 `.ninfer` 模型。纯文字由 NInfer 直接处理；连接图像或视频时，插件会自动使用本机 Qwen3-VL GGUF + mmproj 忠实提取视觉事实，再交给 NInfer 生成最终提示词。这样可避开部分三元 NInfer 转换模型中视觉投影与文本主体语义不对齐而产生的错图描述，无需用户切换模型或额外连线。
- 启动 NInfer 前，插件会自动卸载 ComfyUI 当前驻留的扩散模型、CLIP、VAE，并清理 Torch CUDA 缓存；视觉桥接 GGUF也会先关闭。27B NInfer 几乎需要独占 16GB 显存，这是避免“15.92 GiB / 14.xx GiB”启动失败所必需的。后续图像工作流会由 ComfyUI 按需重新加载原模型。
- 关闭 **启用 NInfer**：选择 `.gguf`；有图像/视频时选择匹配的 `mmproj`。开启 NInfer 时该框自动隐藏。

## 关闭 NInfer 时的引擎

- **GGUF**：直接使用 ComfyUI Python 已安装的 `llama-cpp-python` / `llama.dll`，不启动外部 `llama-server.exe`，也不依赖任何本机外部引擎目录。当前安装的是 CPU 通用版，因此 NVIDIA、AMD、Intel、纯 CPU 都能运行，但速度取决于 CPU；模型会在开启“推理后卸载模型”时释放。

## 输出与上下文限制

- 节点只有一个 `out` 输出口，只输出模型生成的纯文本；除非文字要求明确要求 JSON、分镜或其他格式，否则不会附加解释。
- NInfer 与普通 GGUF 使用同一套最终输出清理：默认移除 `<think>`、`<analysis>`、reasoning 区块以及“最终提示词”前的分析内容。思考模式只影响模型内部推理，不代表把思考过程发送到 `out`；只有用户在“文字要求”中明确要求展示分析或推理过程时才保留。
- 节点不设置固定的生成 token 上限；模型会生成到自然结束、用户文字要求的停止条件，或当前自动上下文的剩余空间耗尽。
- NInfer 与通用 GGUF 后端会根据文本、模板和媒体数量自动选择所需上下文；GGUF 在实际聊天模板比预估更长时会自动扩大后重试。不会向用户暴露固定上下文设置。
- 自动上下文最多跟随模型/运行时支持至 **262144 tokens**。输入、系统提示、TXT/MD、图像/视频视觉 tokens 与输出共享该窗口；显存或内存不足时，运行时会给出真实资源错误而不会静默截断内容。
- GGUF 视觉模型必须与 `mmproj*.gguf` 放在同一目录。
- NInfer 的图像/视频桥接同样需要 `models/LLM` 中至少存在一套 Qwen3-VL（或兼容的 Qwen3.5 视觉）GGUF 与同目录 `mmproj*.gguf`；缺失时会明确报错，纯文字 NInfer 不受影响。

## 一次性配置

NInfer 引擎及运行库已内置于插件的 `engine/ninfer-sm120`（50 系）、`engine/ninfer-sm89`（40 系）和 `engine/runtime` 目录。开启 NInfer 后，插件只使用这些内置文件，不搜索本机其他引擎路径；插件会根据显卡自动选择引擎及 KV 精度，并默认开启思考模式。30 系及其他显卡应关闭 NInfer，使用通用 GGUF 后端。当前随插件提供的 27B NInfer 在本机 RTX 5070 Ti（16GB）上需要 15.92 GiB 连续可用显存，Windows 桌面常驻显存使其无法保证启动；此设备应关闭 NInfer，使用 GGUF 后端。NInfer 服务已经手工运行时插件只复用，绝不关闭未知服务。启动失败时请查看 `engine/ninfer-startup.log`。


视频需要 FFmpeg；若它不在 PATH，请在同一配置文件填写 `ffmpeg.exe` 的完整路径。
