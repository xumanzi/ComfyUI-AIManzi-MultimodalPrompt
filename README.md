# AI蛮子多模态提示词工作台 / AIManzi Multimodal Prompt Workbench

[中文新手教程](#中文新手教程) · [English Guide](#english-guide)

把文字、图片或视频交给 AI，自动生成可以直接用于绘图、视频生成的提示词。既能使用电脑上的本地模型，也能连接 OpenAI 兼容的在线模型。

- [GitHub 完整插件包](https://github.com/xumanzi/ComfyUI-AIManzi-MultimodalPrompt/releases/latest)
- [AI蛮子多模态工作台插件及模型（夸克网盘）](https://pan.quark.cn/s/be99d9acf669)

---

# 中文新手教程

## 这个插件可以做什么？

即使你不会写提示词，也可以用这个插件完成：

- 输入一句简单要求，让模型扩写成详细的绘图提示词。
- 输入一张或多张图片，反推出画面中的人物、环境、光线、构图和风格。
- 输入视频，自动抽取代表画面并生成视频内容提示词。
- 加载 TXT、Markdown、`.skill` 或 Skill ZIP，让模型按照你的模板生成内容。
- 在“本地推理”和“在线推理”之间切换。
- 默认只从 `out` 输出最终提示词，不输出思考过程和多余说明。用户明确要求其他格式时除外。

## 效果截图

### 本地模型

![本地推理工作流](assets/workflow-local.png)

### 在线模型

![在线推理工作流](assets/workflow-online.png)

## 第一步：安装插件

### 方法 A：下载压缩包（最适合新手）

1. 打开 [GitHub Releases](https://github.com/xumanzi/ComfyUI-AIManzi-MultimodalPrompt/releases/latest)。
2. 下载名为 `ComfyUI-AIManzi-MultimodalPrompt-版本号.zip` 的完整插件包。
3. 解压后确认文件夹名称为 `ComfyUI-AIManzi-MultimodalPrompt`。
4. 把整个文件夹复制到：

```text
你的ComfyUI目录\custom_nodes\ComfyUI-AIManzi-MultimodalPrompt
```

5. 完全关闭 ComfyUI，然后重新启动。

> 不要把插件解压成两层同名文件夹。例如 `custom_nodes\插件名\插件名\nodes.py` 是错误的；正确位置应是 `custom_nodes\插件名\nodes.py`。

### 方法 B：使用命令安装

本仓库包含大文件，因此电脑需要先安装 [Git](https://git-scm.com/download/win) 和 [Git LFS](https://git-lfs.com/)。在 PowerShell 中运行：

```powershell
Set-Location "你的ComfyUI目录\custom_nodes"
git lfs install
git clone https://github.com/xumanzi/ComfyUI-AIManzi-MultimodalPrompt.git
Set-Location .\ComfyUI-AIManzi-MultimodalPrompt
git lfs pull
```

以后更新插件：

```powershell
Set-Location "你的ComfyUI目录\custom_nodes\ComfyUI-AIManzi-MultimodalPrompt"
git pull
git lfs pull
```

更新后重新启动 ComfyUI。

## 第二步：准备视频帧

工作台不再接收 `VIDEO` 文件对象，也不直接调用 FFmpeg。请使用 ComfyUI 的加载视频节点先把视频解码成 `IMAGE` 批次，再把它的图像输出连接到“视频帧输入”。如果加载视频节点本身提示缺少 FFmpeg，请按照该加载节点的说明安装依赖；本插件不需要单独配置 FFmpeg 路径。

```text
加载视频节点.图像/IMAGE → 工作台.视频帧输入
```

这样做可以直接使用加载节点输出的连续帧，并避免再次解码视频。

## 第三步：准备本地模型

如果只使用在线推理，可以跳过本节。

把模型文件放入：

```text
你的ComfyUI目录\models\LLM
```

支持的模型：

- `.ninfer`：仅在开启“启用 NInfer”时使用，适合支持的 RTX 40/50 系显卡。
- `.gguf`：关闭 NInfer 后使用，兼容范围更广。
- 视觉 GGUF：要识别图片或视频，主模型和匹配的 `mmproj*.gguf` 必须放在同一文件夹。

本节点不支持直接加载 `.safetensors` 扩散模型或文本编码器。

GGUF 推理还需要在 ComfyUI 自带的 Python 中安装 `llama-cpp-python`。便携版 ComfyUI 示例：

```powershell
Set-Location "你的ComfyUI目录"
& ".\python_embeded\python.exe" -m pip install -U llama-cpp-python
```

如果你的 Python 目录名称不同，请把路径替换成实际的 `python.exe`。

## 第四步：在 ComfyUI 中找到节点

启动 ComfyUI 后，在画布空白位置双击并搜索：

- `AI蛮子 多模态提示词工作台`
- `AI蛮子 加载模板 / Skill`

工作台主要选项：

| 选项 | 小白解释 |
|---|---|
| 文字要求 | 告诉 AI 你想让它做什么 |
| 推理方式 | 选择本地模型或在线 API |
| 模板输入 | 连接“加载模板 / Skill”节点 |
| 图像_1 ～ 图像_10 | 连接图片；连接一个后会自动出现下一个接口 |
| 视频帧输入 | 连接加载视频节点输出的 IMAGE 批次 |
| 视频分析精度 | 快速16帧、标准64帧、高精度128帧、完整逐帧最多256帧 |
| out | 最终生成的纯文字提示词 |

## 使用示例

### 示例 1：只有文字，生成绘图提示词

连接方式：

```text
AI蛮子 多模态提示词工作台.out → Show Text
```

“文字要求”填写：

```text
生成一段东方幻想天宫场景的绘图提示词，包含云海、宫殿、金色晨光和电影级构图。
```

没有连接图片时，插件会按照文字要求直接生成提示词。

### 示例 2：图片反推提示词

连接方式：

```text
加载图像.图像 → 工作台.图像_1
工作台.out → Show Text
```

“文字要求”填写：

```text
观察图片实际内容，反推出可复现该画面的详细正向提示词。
```

可以继续连接更多图片，最多显示 10 个图像接口。图片会在发送前保持比例自动缩放，不会修改原文件。

### 示例 3：视频反推提示词

连接方式：

```text
视频加载节点.图像/IMAGE → 工作台.视频帧输入
工作台.out → Show Text
```

“文字要求”填写：

```text
根据视频中的主体、动作、镜头、环境和光线，只输出一段视频生成提示词。
```

插件按 IMAGE 批次的原始顺序读取视频。短视频会保留全部帧；长视频会同时选择覆盖完整时间线的均匀帧和画面变化明显的关键帧，再按每段最多8帧进行视觉分析，最后汇总动作、运镜、转场、场景和首尾变化。

精度建议：

- 快速：最多16帧，适合测试。
- 标准：最多64帧，适合大多数视频。
- 高精度：最多128帧，适合动作和转场较多的视频。
- 完整逐帧：视频不超过256帧时逐帧分析；更长视频会智能选择最多256帧。耗时和在线费用最高。

### 示例 4：使用模板或 Skill

连接方式：

```text
AI蛮子 加载模板 / Skill.模板输入 → 工作台.模板输入
工作台.out → Show Text
```

点击“上传模板 / Skill”，可以上传 `.txt`、`.md`、`.markdown`、`.skill` 或包含 `SKILL.md` 的 ZIP。插件只读取模板文字，不会执行 Skill 中的脚本。

### 示例 5：在线推理

把“推理方式”切换为“在线推理”，然后填写：

```text
在线_API_URL：https://服务商地址/v1
在线_API_Key：服务商提供的密钥
在线_模型_ID：服务商提供的模型名称
```

在线模式不加载本地 LLM。若需要识别图片或视频，在线模型本身必须支持视觉输入。

> API Key 虽然不会打印到日志，但可能保存在 ComfyUI 工作流文件中。分享工作流前请清空 Key。

## 本地模式应该怎么选？

- RTX 40/50 系并且拥有 `.ninfer` 模型：可以打开 NInfer。
- RTX 30/20 系、AMD、Intel 或纯 CPU：关闭 NInfer，使用 GGUF，或者选择在线推理。
- 需要图片/视频识别：必须使用视觉模型。普通纯文本模型看不懂图片。
- NInfer 识别图片/视频时：插件会先用本地视觉 GGUF + mmproj 读取画面，再交给 NInfer 整理最终提示词。

## 常见问题

### 节点没有出现

检查是否出现了双层插件目录，然后查看 ComfyUI 启动窗口有没有红色报错。修正后必须完整重启 ComfyUI。

### 模型列表是空的

确认模型位于 `ComfyUI\models\LLM`，扩展名是 `.gguf` 或 `.ninfer`，然后重启 ComfyUI。

### 图片接入后仍提示上传图片

必须把“加载图像”节点的 `图像` 输出连接到工作台的 `图像_1`，不是只在文字中填写图片路径。所选模型也必须支持视觉。

### 视频报“未找到 FFmpeg”

这个错误来自负责解码视频的加载视频节点，不是本工作台。请按照该加载节点的安装说明配置 FFmpeg；确认它能够正常输出 IMAGE 批次后，再连接到“视频帧输入”。

### NInfer 启动时显存不足

27B 模型可能需要接近独占 16GB 显存。关闭其他占用显存的软件后重试；仍然无法加载时，关闭 NInfer 并改用 GGUF 或在线推理。降低上下文不能解决模型权重本身装不进显存的问题。

### 为什么输出长度不是无限的？

插件没有固定的 1024-token 输出限制，但最终长度仍受模型上下文、电脑内存/显存和在线服务商限制。输入文字、模板、图片视觉 tokens 与输出共同占用上下文。

---

# English Guide

## What does this plugin do?

AIManzi Multimodal Prompt Workbench turns text, images, videos, and reusable templates into prompts for image or video generation. It supports local NInfer, local GGUF, and OpenAI-compatible online APIs. By default, `out` returns only the final prompt as plain text.

## Beginner installation

1. Download the complete ZIP from [GitHub Releases](https://github.com/xumanzi/ComfyUI-AIManzi-MultimodalPrompt/releases/latest).
2. Extract it to `ComfyUI/custom_nodes/ComfyUI-AIManzi-MultimodalPrompt`.
3. Make sure `nodes.py` is directly inside that folder, not inside a second nested folder.
4. Restart ComfyUI.

Command-line installation requires Git and Git LFS:

```powershell
Set-Location "path-to-ComfyUI\custom_nodes"
git lfs install
git clone https://github.com/xumanzi/ComfyUI-AIManzi-MultimodalPrompt.git
Set-Location .\ComfyUI-AIManzi-MultimodalPrompt
git lfs pull
```

## Video frames

The workbench no longer accepts a `VIDEO` object and does not invoke FFmpeg. Use a ComfyUI video loader to decode the video, then connect its batched `IMAGE` output to `视频帧输入`. If that loader requires FFmpeg, follow the loader's own installation instructions.

## Local models

Place `.ninfer` and `.gguf` files in `ComfyUI/models/LLM`. A vision GGUF requires its matching `mmproj*.gguf` in the same directory. `.safetensors` diffusion models and text encoders are not supported.

Portable ComfyUI GGUF dependency example:

```powershell
Set-Location "path-to-ComfyUI"
& ".\python_embeded\python.exe" -m pip install -U llama-cpp-python
```

## Examples

### Text to prompt

Enter: `Create a cinematic Eastern fantasy palace prompt with clouds and golden morning light.` Connect `out` to a text display node.

### Image to prompt

Connect `Load Image → 图像_1`, then enter: `Describe the visible image as a detailed positive generation prompt.` Up to 10 image inputs can be revealed dynamically.

### Video to prompt

Connect the video loader's batched `IMAGE` output to `视频帧输入`, then enter: `Describe the subjects, motion, camera, environment, and lighting as one video prompt.` Frames remain chronological. Depending on the selected precision, the workbench uses up to 16, 64, 128, or 256 frames, combines uniform timeline coverage with scene-change frames, analyzes chunks of up to eight frames, and synthesizes one final prompt.

### Online inference

Choose online inference and enter the provider's API URL, API key, and model ID. The hosted model must support vision if images or videos are connected. Clear the API key before sharing a workflow because widget values may be serialized.

## Hardware notes

- NInfer is for supported RTX 40/50-series GPUs.
- RTX 30/20-series, AMD, Intel, and CPU-only systems should use GGUF or online inference.
- Local image/video understanding requires a compatible vision GGUF plus mmproj.
- The plugin removes common reasoning blocks by default and does not impose a fixed 1024-token output cap; runtime and provider limits still apply.
