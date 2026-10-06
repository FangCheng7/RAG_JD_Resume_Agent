# -*- mode: python ; coding: utf-8 -*-
"""
TalentMatch AI — PyInstaller 打包配置

用法：
    pyinstaller --noconfirm --clean TalentMatchAI.spec

产物：
    dist/TalentMatchAI/TalentMatchAI.exe

为什么用「目录模式」而不是单文件（onefile）？
    torch + gradio 加起来有好几个 GB，
    单文件模式每次启动都要把它们解压一遍，
    要等好几分钟才能打开，完全没法用。
    目录模式只在第一次解压，之后直接加载，启动只要十几秒。
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_data_files


# ============================================================
# 需要连「数据文件 + 子模块」一起收进来的包
# ============================================================
#
# 这些包都有运行时动态加载的资源：
#   gradio  → 前端 HTML / JS / CSS 静态文件
#   chromadb → 内置的 onnx 模型与迁移脚本
#   tiktoken → BPE 词表
#   transformers / FlagEmbedding → 配置与 tokenizer 定义
#
# 不显式收集的话，打包出来会报「找不到文件」。

COLLECT_PACKAGES = [
    "gradio",
    "gradio_client",
    "safehttpx",
    "groovy",
    "chromadb",
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_chroma",
    "langchain_text_splitters",
    "langgraph",
    "langgraph_checkpoint",
    "langgraph_prebuilt",
    "openai",
    "tiktoken",
    "transformers",
    "sentence_transformers",
    "FlagEmbedding",
    "pypdf",
    "docx",
    "huggingface_hub",
]

datas = []
binaries = []
hiddenimports = []

for package in COLLECT_PACKAGES:

    try:

        d, b, h = collect_all(package)

        datas += d
        binaries += b
        hiddenimports += h

    except Exception as exc:

        # 没装的包直接跳过，不影响其他部分
        print(f"[spec] 跳过 {package}: {exc}")

# Gradio 的前端资源再单独兜一遍，双保险
datas += collect_data_files("gradio", include_py_files=False)


# ============================================================
# 入口
# ============================================================

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],

    # 排除用不到的东西，能省 250MB+ 体积。
    #
    # 下面这几个大包都做过实测：
    # 导入 src.ui（会把 agent / tools / vectorstore 全带起来）
    # 之后它们【完全没有被加载】，所以可以安全剔除。
    excludes=[
        # GUI / 绘图，本来就用不到
        "tkinter",
        "matplotlib",
        "PIL.ImageQt",

        # ---- 以下实测未被加载 ----
        "spacy",        # 90 MB —— 早就不用的 unstructured 的依赖
        "thinc",        # spacy 的依赖
        "blis",         # spacy 的依赖
        "numba",        # pandas 的可选加速，不走它
        "llvmlite",     # 115 MB —— numba 的依赖
        "onnxruntime",  # 36 MB —— chromadb 的可选 embedding，我们用 BGE-M3
        "unstructured", # 已改用 pypdf + python-docx
        "nltk",         # unstructured 的依赖
    ],

    noarchive=False,
    optimize=0,
)

# ============================================================
# 关键修复：剔除 PyInstaller 自动打包的 VC 运行库
# ============================================================
#
# 踩过的坑：
#
#   PyInstaller 会把自己找到的 msvcp140.dll / vcruntime140.dll
#   一起放进 _internal 目录，但它们的版本往往比系统自带的旧。
#
#   而 torch 需要较新的 VC 运行库，
#   旧的把新的遮住之后，加载 DLL 就会失败：
#
#       OSError: [WinError 1114] 动态链接库(DLL)初始化例程失败
#         Error loading "...\torch\lib\c10.dll" or one of its dependencies
#
#   表现就是：双击 exe 闪一下就没了，服务起不来。
#
# 解决办法：
#
#   不打包这几个 DLL，让程序直接使用系统自带的
#   （torch 本来就要求机器上装了 VC++ 运行库）。
#
# 注意：VCOMP140.DLL 要保留，它是 OpenMP 运行库，
#       且与系统版本一致，留着不会有冲突。

EXCLUDE_VC_DLLS = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_atomic_wait.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
}

# 只剔除「顶层」的那几个 DLL，也就是 _internal\msvcp140.dll 这种。
#
# 注意不要连子目录里的一起删：
#   sklearn\.libs\msvcp140.dll 是 scikit-learn 自带的，
#   它必须留在原位，否则会报
#       Failed to load dynlib/dll '...\sklearn\.libs\msvcp140.dll'
a.binaries = [
    entry
    for entry in a.binaries
    if not (
        os.path.basename(entry[0]).lower() in EXCLUDE_VC_DLLS
        and os.path.dirname(entry[0]) in ("", ".")
    )
]


# ============================================================
# 双保险：把用不到的大包从 binaries / datas 里清掉
# ============================================================
#
# Analysis 的 excludes 只作用于「模块依赖图」，
# 但 collect_all() 是把文件直接塞进 datas / binaries 的，
# 绕过了依赖图。所以这里再按目录名过滤一遍。

EXCLUDE_PACKAGE_DIRS = {
    "spacy",
    "thinc",
    "blis",
    "numba",
    "llvmlite",
    "onnxruntime",
    "unstructured",
    "nltk",
}


def _in_excluded_package(dest):
    """判断这个文件是不是属于要剔除的包。"""

    parts = str(dest).replace("\\", "/").split("/")

    return any(part.lower() in EXCLUDE_PACKAGE_DIRS for part in parts)


_before = len(a.binaries) + len(a.datas)

a.binaries = [e for e in a.binaries if not _in_excluded_package(e[0])]
a.datas = [e for e in a.datas if not _in_excluded_package(e[0])]

print(
    f"[spec] 剔除无用包：文件数 {_before} -> "
    f"{len(a.binaries) + len(a.datas)}"
)


pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="TalentMatchAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,

    # 保留控制台窗口：
    # 启动后会在里面打印访问地址，
    # 出错时也能直接看到原因。
    console=True,

    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="TalentMatchAI",
)