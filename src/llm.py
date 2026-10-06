"""
TalentMatch AI
LLM 模块

负责：

1. 创建 DeepSeek LLM
2. 从本地配置读取 API
3. 避免重复创建 LLM
4. 用户修改 API 后清除缓存

DeepSeek 使用 OpenAI-compatible API，
所以这里使用 LangChain 的 ChatOpenAI。
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import (
    get_api_key,
    get_base_url,
    get_model,
)


# ============================================================
# 创建 LLM
# ============================================================

@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    创建 DeepSeek LLM。

    使用 @lru_cache 是为了：

        第一次调用
            ↓
        创建 LLM
            ↓
        后续重复使用

    而不是每一次 Agent 请求都重新创建模型。

    Returns:
        ChatOpenAI:
            LangChain LLM 对象。

    Raises:
        RuntimeError:
            用户还没有配置 API Key。
    """

    api_key = get_api_key()
    base_url = get_base_url()
    model = get_model()

    # 用户还没有输入 API Key。
    if not api_key:

        raise RuntimeError(
            "尚未配置 API Key。\n"
            "请进入「系统设置」输入 API Key 并保存。"
        )

    # DeepSeek 兼容 OpenAI API。
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,

        # 招聘匹配属于结构化任务，
        # 温度设为 0 可以减少随机性。
        temperature=0,

        # 单次请求最多等待 120 秒。
        timeout=120,

        # 网络临时失败时自动重试。
        max_retries=3,
    )


# ============================================================
# 清除 LLM 缓存
# ============================================================

def clear_llm_cache():
    """
    清除 LLM 缓存。

    当用户在网页中修改：

    API Key
    Base URL
    Model

    后，需要清除旧 LLM，
    这样下一次调用 get_llm()
    才会重新读取新配置。
    """

    get_llm.cache_clear()