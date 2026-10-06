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

    # 明显用不到的 GUI / 绘图库，排除掉能省不少体积
    excludes=[
        "tkinter",
        "matplotlib",
        "PIL.ImageQt",
    ],

    noarchive=False,
    optimize=0,
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