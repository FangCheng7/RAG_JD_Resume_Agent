"""
TalentMatch AI
Agent Tools

这里定义 Agent 可以调用的工具。

整个 Agent 并不是直接把所有代码写死：

用户
 ↓
Agent
 ↓
判断需要什么工具
 ↓
调用 Tool
 ↓
得到结果
 ↓
继续推理

当前工具：

1. search_documents
   RAG 检索

2. get_document
   获取完整文档

3. list_documents
   查看知识库文档

4. jd_parse_tool
   解析 JD

5. resume_parse_tool
   解析简历

6. match_tool
   JD / Resume 匹配
"""

import json
from typing import Any

from langchain_core.tools import tool

from src.config import (
    MAX_DOCUMENT_CHARS,
    TOP_K,
)

from src.llm import get_llm

from src.vectorstore import (
    get_vectorstore,
)


# ============================================================
# 辅助函数
# ============================================================

def _content_to_text(
    content: Any,
) -> str:
    """
    将 LangChain 返回的各种 content
    统一转换为字符串。
    """

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                if item.get(
                    "type"
                ) == "text":

                    parts.append(
                        item.get(
                            "text",
                            "",
                        )
                    )

            elif isinstance(
                item,
                str,
            ):

                parts.append(
                    item
                )

        return "\n".join(
            parts
        )

    return str(content)


def _extract_json(
    text: str,
) -> dict:
    """
    从 LLM 输出中提取 JSON。

    有时候模型会返回：

    ```json
    {...}
    ```

    所以这里需要把 Markdown 包装去掉。
    """

    text = text.strip()

    if text.startswith(
        "```"
    ):

        text = (
            text
            .replace(
                "```json",
                "",
            )
            .replace(
                "```",
                "",
            )
            .strip()
        )

    start = text.find(
        "{"
    )

    end = text.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
    ):

        raise ValueError(
            "LLM 没有返回有效 JSON。"
        )

    return json.loads(
        text[
            start:end + 1
        ]
    )


def _json_llm_call(
    system_prompt: str,
    user_prompt: str,
) -> dict:
    """
    调用 DeepSeek，并要求返回 JSON。
    """

    llm = get_llm()

    response = llm.invoke(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
    )

    content = _content_to_text(
        response.content
    )

    return _extract_json(
        content
    )


# ============================================================
# Tool 1：RAG 搜索
# ============================================================

@tool
def search_documents(
    query: str,
    doc_type: str = "both",
    k: int = TOP_K,
) -> str:
    """
    搜索 JD / Resume。

    这是整个 RAG 的 Retrieval 阶段。

    用户问题
        ↓
    BGE-M3
        ↓
    Query Vector
        ↓
    Chroma
        ↓
    Top-K 相关 Chunk
    """

    store = get_vectorstore()

    search_k = max(
        1,
        min(
            int(k),
            30,
        ),
    )

    # --------------------------------------------------------
    # 只搜索 JD
    # --------------------------------------------------------

    if doc_type == "jd":

        docs = store.similarity_search(
            query,
            k=search_k,
            filter={
                "doc_type": "jd"
            },
        )

    # --------------------------------------------------------
    # 只搜索简历
    # --------------------------------------------------------

    elif doc_type == "resume":

        docs = store.similarity_search(
            query,
            k=search_k,
            filter={
                "doc_type": "resume"
            },
        )

    # --------------------------------------------------------
    # 搜索全部
    # --------------------------------------------------------

    else:

        docs = store.similarity_search(
            query,
            k=search_k,
        )

    if not docs:

        return (
            "没有找到相关文档。"
        )

    results = []

    for index, doc in enumerate(
        docs,
        1,
    ):

        results.append(
            f"""
--- Result {index} ---

文档类型：
{doc.metadata.get("doc_type")}

文件名：
{doc.metadata.get("filename")}

document_id：
{doc.metadata.get("doc_id")}

chunk：
{doc.metadata.get("chunk_index")}

内容：
{doc.page_content}
"""
        )

    return "\n".join(
        results
    )


# ============================================================
# Tool 2：获取完整文档
# ============================================================

@tool
def get_document(
    doc_id: str,
) -> str:
    """
    根据 document_id 获取完整文档。

    RAG 检索通常先找到某个 Chunk，
    如果 Agent 认为需要完整上下文，
    可以进一步调用这个 Tool。
    """

    store = get_vectorstore()

    result = store.get(
        where={
            "doc_id": doc_id
        },
        include=[
            "documents",
            "metadatas",
        ],
    )

    documents = result.get(
        "documents",
        [],
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    if not documents:

        return (
            f"没有找到 document_id："
            f"{doc_id}"
        )

    items = []

    for document, metadata in zip(
        documents,
        metadatas,
    ):

        items.append(
            (
                metadata.get(
                    "chunk_index",
                    0,
                ),
                document,
            )
        )

    # 按 Chunk 顺序重新组合。
    items.sort(
        key=lambda x: x[0]
    )

    text = "\n\n".join(
        item[1]
        for item in items
    )

    # 防止一次性把超长简历送给 LLM。
    return text[
        :MAX_DOCUMENT_CHARS
    ]


# ============================================================
# Tool 3：列出文档
# ============================================================

@tool
def list_documents(
    doc_type: str = "both",
) -> str:
    """
    查看知识库中有哪些 JD / 简历。
    """

    store = get_vectorstore()

    result = store.get(
        include=[
            "metadatas"
        ]
    )

    metadatas = result.get(
        "metadatas",
        [],
    )

    documents = {}

    for metadata in metadatas:

        current_type = metadata.get(
            "doc_type"
        )

        # 如果指定类型，
        # 就过滤。
        if (
            doc_type != "both"
            and current_type != doc_type
        ):
            continue

        doc_id = metadata.get(
            "doc_id"
        )

        if not doc_id:
            continue

        # 一个文档有多个 Chunk，
        # 所以需要按照 doc_id 去重。
        if doc_id not in documents:

            documents[doc_id] = {
                "doc_id": doc_id,
                "doc_type": current_type,
                "filename": metadata.get(
                    "filename"
                ),
            }

    if not documents:

        return "当前没有文档。"

    lines = []

    for item in documents.values():

        lines.append(
            f"""
document_id:
{item["doc_id"]}

类型：
{item["doc_type"]}

文件名：
{item["filename"]}
"""
        )

    return "\n".join(
        lines
    )


# ============================================================
# Tool 4：JD 解析
# ============================================================

@tool
def jd_parse_tool(
    jd_text: str,
) -> str:
    """
    使用 DeepSeek 将 JD 转换为结构化 JSON。

    这一步属于：

    非结构化文本
        ↓
    LLM Information Extraction
        ↓
    Structured Data
    """

    system_prompt = """
你是一名专业招聘分析专家。

请把岗位 JD 转换成结构化 JSON。

只能根据 JD 中明确出现的信息提取。

不要猜测。

不要根据年龄、性别、民族、宗教、
婚姻状况、残疾等属性进行分析。

JSON 格式：

{
  "job_title": "",
  "department": "",
  "hard_requirements": [],
  "skills": [],
  "responsibilities": [],
  "education": "",
  "years_of_experience": ""
}
"""

    user_prompt = f"""
请解析下面的 JD：

{jd_text}
"""

    result = _json_llm_call(
        system_prompt,
        user_prompt,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Tool 5：简历解析
# ============================================================

@tool
def resume_parse_tool(
    resume_text: str,
) -> str:
    """
    使用 DeepSeek 将简历转换成结构化数据。
    """

    system_prompt = """
你是一名专业招聘分析专家。

请把候选人简历转换成结构化 JSON。

只能根据简历中明确出现的信息。

不要猜测。

禁止使用：

年龄
性别
民族
宗教
婚姻状况
残疾

JSON：

{
  "name": "",
  "years_of_experience": "",
  "education": [],
  "skills": [],
  "work_experience": [],
  "project_experience": []
}
"""

    user_prompt = f"""
请解析下面的简历：

{resume_text}
"""

    result = _json_llm_call(
        system_prompt,
        user_prompt,
    )

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Tool 6：JD / Resume 匹配
# ============================================================

@tool
def match_tool(
    jd_data: str,
    resumes_data: str,
) -> str:
    """
    对 JD 和多个候选人进行匹配。

    输出：

    score
    match_points
    weaknesses
    evaluation
    """

    system_prompt = """
你是一名专业招聘评估 Agent。

根据岗位要求和候选人信息进行岗位匹配。

评分范围：

0-100。

重点考虑：

- 岗位技能
- 工作经验
- 项目经验
- 教育背景
- JD 硬性要求
- 岗位职责匹配度

禁止根据：

- 年龄
- 性别
- 民族
- 宗教
- 婚姻状况
- 残疾
- 其他受保护属性

进行评分。

如果简历没有提供某项信息，
不能直接认为候选人没有该能力。

返回：

{
  "candidates": [
    {
      "name": "",
      "score": 0,
      "match_points": [],
      "weaknesses": [],
      "evaluation": ""
    }
  ]
}
"""

    user_prompt = f"""
岗位：

{jd_data}

候选人：

{resumes_data}
"""

    result = _json_llm_call(
        system_prompt,
        user_prompt,
    )

    candidates = result.get(
        "candidates",
        [],
    )

    # LLM 输出后再由 Python 排序，
    # 而不是完全相信模型排序。
    candidates.sort(
        key=lambda x: x.get(
            "score",
            0,
        ),
        reverse=True,
    )

    result["candidates"] = candidates

    return json.dumps(
        result,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# Agent Tool List
# ============================================================

TOOLS = [
    search_documents,
    get_document,
    list_documents,
    jd_parse_tool,
    resume_parse_tool,
    match_tool,
]