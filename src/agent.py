"""
TalentMatch AI
Agent 核心

这是项目最重要的部分之一。

传统程序：

用户
 ↓
固定流程
 ↓
固定结果

Agent：

用户
 ↓
LLM
 ↓
判断需要什么工具
 ↓
调用 Tool
 ↓
观察结果
 ↓
继续调用 Tool
 ↓
最终回答

这里使用 LangGraph / LangChain Agent
实现 Tool Calling。
"""

from functools import lru_cache

from langchain.agents import create_agent

from src.llm import (
    clear_llm_cache,
    get_llm,
)

from src.tools import TOOLS


# ============================================================
# Agent System Prompt
# ============================================================

SYSTEM_PROMPT = """
你是 TalentMatch AI。

你是一个企业招聘 JD / 简历智能匹配 Agent。

你的任务是帮助招聘人员：

1. 查询 JD
2. 查询候选人简历
3. 解析 JD
4. 解析简历
5. 分析岗位要求
6. 计算候选人与岗位的匹配度
7. 对候选人排序
8. 对比候选人
9. 生成招聘分析报告

你可以调用以下工具：

search_documents
get_document
list_documents
jd_parse_tool
resume_parse_tool
match_tool


============================================================
候选人筛选流程
============================================================

当用户说：

「帮我筛选最匹配的5个人」

应该：

1. 搜索 JD
2. 获取 JD
3. 解析 JD
4. 搜索候选人简历
5. 获取候选人简历
6. 解析候选人
7. 调用 match_tool
8. 排序
9. 输出结果


============================================================
简历找岗位流程
============================================================

当用户说：

「这份简历适合什么岗位？」

应该：

1. 获取简历
2. 解析简历
3. 搜索相关 JD
4. 分别进行匹配
5. 给出推荐岗位


============================================================
候选人对比
============================================================

当用户说：

「比较候选人 A 和 B」

应该：

1. 找到候选人
2. 找到目标 JD
3. 分析两个候选人
4. 调用匹配工具
5. 对比优势和不足


============================================================
重要原则
============================================================

不要编造信息。

如果数据库没有相关信息，
明确告诉用户。

不要把「没有提供」理解为「没有能力」。

不要根据：

年龄
性别
民族
宗教
婚姻状况
残疾

等受保护属性进行评分。

匹配分数只是辅助决策。

最终招聘决定必须由人工完成。

回答使用中文。

最终结果应该：

结构化
简洁
清晰
适合招聘人员阅读
"""


# ============================================================
# Agent 创建
# ============================================================

@lru_cache(maxsize=1)
def get_agent():
    """
    创建 TalentMatch Agent。

    Agent 包含：

    LLM
    +
    Tools
    +
    System Prompt
    """

    llm = get_llm()

    return create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="talentmatch_agent",
    )


# ============================================================
# 清理 Agent 缓存
# ============================================================

def clear_agent_cache():
    """
    用户修改 API 后，
    清除旧 Agent。

    下一次调用 get_agent()
    会重新创建。
    """

    get_agent.cache_clear()


# ============================================================
# 对外聊天接口
# ============================================================

def _normalize_history(history) -> list[dict]:
    """
    把 Gradio 传来的对话历史转换成 LangChain 的 messages。

    Gradio 有两种格式：

        1. messages 格式（新版）：
           [{"role": "user", "content": "..."}, ...]

        2. tuples 格式（旧版）：
           [["用户问题", "AI 回答"], ...]

    这里两种都做兼容。
    """

    messages: list[dict] = []

    if not history:
        return messages

    for item in history:

        # ------------------------------------------------
        # 格式一：{"role": ..., "content": ...}
        # ------------------------------------------------

        if isinstance(item, dict):

            role = item.get("role")
            content = item.get("content")

            if role in ("user", "assistant") and content:

                messages.append(
                    {
                        "role": role,
                        "content": str(content),
                    }
                )

            continue

        # ------------------------------------------------
        # 格式二：["用户问题", "AI 回答"]
        # ------------------------------------------------

        if isinstance(item, (list, tuple)) and len(item) == 2:

            user_text, assistant_text = item

            if user_text:

                messages.append(
                    {
                        "role": "user",
                        "content": str(user_text),
                    }
                )

            if assistant_text:

                messages.append(
                    {
                        "role": "assistant",
                        "content": str(assistant_text),
                    }
                )

    return messages


def _extract_text(message_obj) -> str:
    """
    从 LangChain 的消息对象中提取可显示的文本。

    不同模型 / 版本返回的 content 可能是：

        str
        list[dict]
        list[str]
    """

    content = getattr(
        message_obj,
        "content",
        "",
    )

    if isinstance(content, str):

        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                parts.append(item)

        return "\n".join(parts).strip()

    return str(content).strip()


def chat(
    message: str,
    history=None,
    context: str | None = None,
) -> str:
    """
    给 UI 使用的聊天接口。

    Gradio：

        用户输入 + 历史对话
            ↓
        chat()
            ↓
        Agent
            ↓
        Tools
            ↓
        最终回答

    支持多轮对话：
    历史消息会一起交给 Agent，
    因此「对比上面两个候选人」这类问题也能理解上下文。

    context：
        可选的「限定范围」说明。

        前端选择了具体的岗位 / 简历后，
        会把它们的 document_id 作为 context 传进来，
        这样 Agent 就只围绕这些文档工作，
        而不是在整个知识库里盲搜。
    """

    if not message or not message.strip():

        return "请输入招聘任务。"

    try:

        agent = get_agent()

        # 组装 messages：历史 + 当前问题
        messages = _normalize_history(history)

        # 如果前端限定了岗位 / 简历，
        # 就把范围说明拼在本轮问题前面。
        #
        # 这样做的好处是：
        #   1. 历史记录里仍然是干净的原始问题
        #   2. 只有本轮请求会带上限定范围
        #   3. 不依赖模型是否遵守 system 消息的位置
        user_content = message

        if context and context.strip():

            user_content = (
                f"{context.strip()}\n\n"
                "===== 用户问题 =====\n"
                f"{message}"
            )

        messages.append(
            {
                "role": "user",
                "content": user_content,
            }
        )

        result = agent.invoke(
            {
                "messages": messages
            }
        )

        result_messages = result.get(
            "messages",
            [],
        )

        if not result_messages:

            return "Agent 没有返回结果。"

        # 从最后向前寻找真正的文本回答。
        for message_obj in reversed(result_messages):

            text = _extract_text(message_obj)

            if text:

                return text

        return "Agent 没有生成可显示的结果。"

    except Exception as exc:

        # 不让前端直接出现 Python Traceback。
        kind = type(exc).__name__

        # 网络类错误时，把缓存的 LLM / Agent 一起清掉。
        #
        # 原因：
        #   LangChain 复用的是同一个 OpenAI 客户端，
        #   而它内部维护连接池。
        #   一旦发生连接错误，连接池可能卡在坏状态，
        #   后续请求会一直失败。
        #   清掉缓存后，下一次请求会重建一个干净的客户端。
        is_network_error = any(
            key in kind
            for key in (
                "Connection",
                "Timeout",
                "Connect",
                "SSL",
                "Proxy",
            )
        )

        if is_network_error:

            try:

                clear_llm_cache()
                clear_agent_cache()

            except Exception:

                pass

        # 把底层原因也挖出来，
        # 否则只看到一句 "Connection error." 无法排查。
        root = exc

        for _ in range(6):

            nxt = (
                getattr(root, "__cause__", None)
                or getattr(root, "__context__", None)
            )

            if nxt is None or nxt is root:

                break

            root = nxt

        message = (
            "Agent 运行失败。\n\n"
            f"错误类型：{kind}\n\n"
            f"错误信息：{exc}"
        )

        if root is not exc:

            message += (
                "\n\n**底层原因：** "
                f"`{type(root).__name__}: {root}`"
            )

        if is_network_error:

            message += (
                "\n\n---\n\n"
                "这通常是**网络**问题，可以这样排查：\n\n"
                "1. 确认电脑能正常上网\n"
                "2. 浏览器打开 "
                "https://api.deepseek.com 看能否访问\n"
                "3. 到「系统设置」点一下 **测试连接**\n"
                "4. 等几秒后**重新发送**（连接已自动重建）\n\n"
                "如果一直失败，也可能是 DeepSeek 服务临时波动，"
                "稍后再试即可。"
            )

        return message
