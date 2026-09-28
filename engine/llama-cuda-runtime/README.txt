AI蛮子 GGUF CUDA 运行库目录

插件会优先从本目录加载 llama.cpp 所需的 NVIDIA CUDA 运行库。
发布包可以把 NVIDIA 官方允许再分发的 CUDA 13 DLL 放在这里。

为避免插件重复携带约 500 MB 的 cuBLAS 文件，默认还会自动复用：
1. ComfyUI Python 自带的 nvidia/cu13 运行库；
2. 本机 CUDA Toolkit；
3. NVIDIA 显卡驱动 DriverStore 中的 nvcudart_hybrid64.dll。

插件不会从第三方 DLL 网站下载文件，也不会修改 Windows 系统目录。
