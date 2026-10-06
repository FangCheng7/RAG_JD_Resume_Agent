"""
TalentMatch AI
配置管理模块

这个文件负责整个项目的基础配置。

主要职责：

1. 确定项目目录
2. 创建 data / chroma / uploads 等目录
3. 读取环境变量
4. 读取用户在网页「系统设置」中保存的 API 配置
5. 保存用户 API 配置
6. 管理 RAG 相关参数

配置优先级：

网页系统设置 / data/config.json
        ↓
.env
        ↓
代码默认值
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# 1. 加载 .env
# ============================================================

# 读取项目根目录下的 .env。
#
# .env 主要作为开发阶段的备用配置。
# 正式使用时，普通用户可以直接通过网页「系统设置」配置 API。
load_dotenv()


# ============================================================
# 2. 项目目录
# ============================================================

# __file__：
# 当前文件 src/config.py
#
# parent：
# src/
#
# parent.parent：
# 项目根目录 rag_jd_resume_agent/
# 判断当前是否运行在 PyInstaller 打包出来的 exe 里。
#
# 打包后 __file__ 指向的是解压出来的临时目录，
# 用它算路径会把 data / models 写进临时目录里
# （关掉程序就没了，用户也看不到）。
#
# 所以打包环境下用 exe 自己所在的目录，
# 这样 data / models 就放在 exe 旁边，用户看得见、也能替换。
if getattr(sys, "frozen", False):

    BASE_DIR = Path(sys.executable).resolve().parent

else:

    BASE_DIR = Path(__file__).resolve().parent.parent


# 数据目录
DATA_DIR = BASE_DIR / "data"

# Chroma 本地向量数据库
CHROMA_DIR = DATA_DIR / "chroma"

# 用户上传的 JD / 简历
UPLOAD_DIR = DATA_DIR / "uploads"

# 用户通过网页保存的 API 配置
CONFIG_FILE = DATA_DIR / "config.json"

# 本地 Embedding 模型目录
#
# 运行 download_model.py 后，
# BGE-M3 会存放在这里。
#
# 使用本地目录的好处：
#
#   1. 不需要联网
#   2. 绕开 huggingface.co 在国内无法访问的问题
#   3. 启动更快
LOCAL_EMBEDDING_DIR = BASE_DIR / "models" / "bge-m3"


# ============================================================
# 3. 自动创建目录
# ============================================================

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 4. 默认 API 配置
# ============================================================

DEFAULT_CONFIG = {
    # 优先读取 .env。
    # 如果没有，就使用空字符串。
    "api_key": os.getenv(
        "DEEPSEEK_API_KEY",
        "",
    ),

    # DeepSeek OpenAI-compatible API 地址
    "base_url": os.getenv(
        "DEEPSEEK_BASE_URL",
        "https://api.deepseek.com",
    ),

    # 默认使用 DeepSeek Chat
    "model": os.getenv(
        "DEEPSEEK_MODEL",
        "deepseek-chat",
    ),
}


# ============================================================
# 5. 用户配置读取
# ============================================================

def load_user_config() -> dict:
    """
    读取用户在网页「系统设置」保存的配置。

    配置文件：

        data/config.json

    如果用户还没有保存过配置，
    就使用 DEFAULT_CONFIG。

    Returns:
        dict:
            包含 api_key、base_url、model。
    """

    # 第一次启动时 config.json 可能不存在。
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()

    try:

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        # 从默认配置开始。
        config = DEFAULT_CONFIG.copy()

        # 如果 JSON 确实是字典，
        # 就覆盖默认值。
        if isinstance(data, dict):
            config.update(data)

        return config

    except Exception:

        # 如果配置文件损坏，
        # 不让整个程序直接崩溃。
        return DEFAULT_CONFIG.copy()


# ============================================================
# 6. 保存用户配置
# ============================================================

def save_user_config(
    api_key: str,
    base_url: str,
    model: str,
) -> dict:
    """
    保存用户在网页中输入的 API 配置。

    注意：
    API Key 会保存到本机：

        data/config.json

    该文件必须加入 .gitignore，
    防止 API Key 被上传到 GitHub。

    Args:
        api_key:
            DeepSeek API Key

        base_url:
            API Base URL

        model:
            使用的模型名称

    Returns:
        保存后的配置字典。
    """

    config = {
        "api_key": api_key.strip(),
        "base_url": base_url.strip(),
        "model": model.strip(),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            config,
            file,
            ensure_ascii=False,
            indent=4,
        )

    return config


# ============================================================
# 7. 单独读取配置
# ============================================================

def get_api_key() -> str:
    """
    获取当前 API Key。
    """

    return load_user_config().get(
        "api_key",
        "",
    )


def get_base_url() -> str:
    """
    获取当前 API Base URL。
    """

    return load_user_config().get(
        "base_url",
        "https://api.deepseek.com",
    )


def get_model() -> str:
    """
    获取当前 LLM 模型名称。
    """

    return load_user_config().get(
        "model",
        "deepseek-chat",
    )


# ============================================================
# 8. Embedding 配置
# ============================================================

# BGE-M3 在本地运行。
#
# 它负责：
#
# 文本
#  ↓
# Embedding Vector
#
# 然后向量进入 Chroma。
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-m3",
)


# ============================================================
# 9. RAG 参数
# ============================================================

# 一个 Chunk 最大字符数量。
CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "900",
    )
)


# Chunk 之间重叠的字符数量。
#
# 重叠可以减少因为切分导致上下文断裂。
CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "150",
    )
)


# 默认检索多少个 Chunk。
TOP_K = int(
    os.getenv(
        "TOP_K",
        "12",
    )
)


# 单个完整文档最多送给 LLM 多少字符。
MAX_DOCUMENT_CHARS = int(
    os.getenv(
        "MAX_DOCUMENT_CHARS",
        "50000",
    )
)