"""
services.py
============================================================

TalentMatch AI 的业务服务层。

这一层负责：

1. 给 Dashboard 提供统计数据
2. 给前端提供文档列表
3. 对底层 Chroma 数据进行业务层封装
4. 删除文档（向量 + 本地原文件）

为什么需要 services.py？

因为我们不希望 ui.py 直接操作 Chroma。

推荐的项目结构是：

    UI
     ↓
    services
     ↓
    vectorstore
     ↓
    Chroma

这样做的好处是：

- UI 只负责界面
- services 负责业务逻辑
- vectorstore 负责数据库
- 后续如果换数据库，UI 不需要大改

这也是一种比较标准的 AI 应用分层方式。

注意：

    统计 / 列表 / 删除这类操作
    并不需要把文本转成向量。

    所以这里直接使用轻量的 chromadb 客户端，
    避免为了显示一个数字而加载 BGE-M3 模型
    （那会让首页打开非常慢）。
"""

from pathlib import Path
from typing import Any

import chromadb

from src.config import CHROMA_DIR, UPLOAD_DIR


# ============================================================
# Chroma 常量
# ============================================================

# 必须和 vectorstore.py 中的 collection_name 保持一致。
COLLECTION_NAME = "jd_resume_chunks"


# ============================================================
# 底层：获取 Collection
# ============================================================

def _get_collection():
    """
    以「只读优先」的方式获取 Chroma Collection。

    如果知识库还不存在，返回 None。

    这里没有使用 get_vectorstore()，
    因为 get_vectorstore() 会连带加载 BGE-M3，
    对于统计 / 列表 / 删除来说没有必要。
    """

    try:

        client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        return client.get_collection(
            COLLECTION_NAME
        )

    except Exception:

        # Collection 还不存在，
        # 或者数据库暂时不可用。
        return None


def _load_metadatas() -> list[dict[str, Any]]:
    """
    读取所有 Chunk 的 metadata。

    一个文档会被切成多个 Chunk，
    因此这里的长度是 chunk 数量，
    而不是文档数量。
    """

    collection = _get_collection()

    if collection is None:
        return []

    try:

        data = collection.get(
            include=["metadatas"]
        )

    except Exception:

        return []

    metadatas = data.get("metadatas") or []

    return [
        metadata
        for metadata in metadatas
        if metadata
    ]


def _collect_documents(
    metadatas: list[dict[str, Any]],
    doc_type: str = "both",
) -> list[dict[str, Any]]:
    """
    把 Chunk 级别的 metadata 聚合为「文档级别」的列表。

    聚合规则：

        一个 doc_id → 一个文档
        同一个 doc_id 的多个 chunk 只算一次

    同时统计每个文档包含多少个 chunk。
    """

    documents: dict[str, dict[str, Any]] = {}

    for metadata in metadatas:

        current_type = metadata.get("doc_type")

        # 类型过滤
        if (
            doc_type != "both"
            and current_type != doc_type
        ):
            continue

        doc_id = metadata.get("doc_id")

        if not doc_id:
            continue

        if doc_id not in documents:

            documents[doc_id] = {
                "doc_id": doc_id,
                "filename": metadata.get(
                    "filename",
                    "未命名文件",
                ),
                "doc_type": current_type,
                "saved_filename": metadata.get(
                    "saved_filename",
                    "",
                ),
                "chunk_count": 0,
            }

        documents[doc_id]["chunk_count"] += 1

    result = list(documents.values())

    result.sort(
        key=lambda item: (
            str(item.get("doc_type", "")),
            str(item.get("filename", "")),
        )
    )

    return result


# ============================================================
# Dashboard 统计
# ============================================================

def get_dashboard_stats() -> dict[str, Any]:
    """
    获取首页 Dashboard 所需要的统计数据。

    返回：

        {
            "jd_count": JD 数量,
            "resume_count": 简历数量,
            "document_count": 总文档数量,
            "chunk_count": 文本块数量（同时也是向量数量）
        }

    注意：

        一个 PDF / DOCX 会被切分成多个 chunk。
        因此 document_count 和 chunk_count 是两个不同的值。
    """

    metadatas = _load_metadatas()

    documents = _collect_documents(metadatas)

    jd_count = sum(
        1
        for item in documents
        if item["doc_type"] == "jd"
    )

    resume_count = sum(
        1
        for item in documents
        if item["doc_type"] == "resume"
    )

    return {
        "jd_count": jd_count,
        "resume_count": resume_count,
        "document_count": len(documents),
        "chunk_count": len(metadatas),

        # 兼容旧字段名
        "jd_documents": jd_count,
        "resume_documents": resume_count,
        "total_documents": len(documents),
        "total_chunks": len(metadatas),
    }


# ============================================================
# 文档列表
# ============================================================

def get_documents(
    doc_type: str = "both",
) -> list[dict[str, Any]]:
    """
    获取系统中的文档列表。

    参数：

        doc_type

    可以是：

        "jd"
        "resume"
        "both"

    返回的数据大概是：

        [
            {
                "doc_id": "...",
                "filename": "AI产品经理.pdf",
                "doc_type": "jd",
                "saved_filename": "...",
                "chunk_count": 12
            }
        ]
    """

    metadatas = _load_metadatas()

    return _collect_documents(
        metadatas,
        doc_type,
    )


# ============================================================
# 删除文档
# ============================================================

def delete_document(
    doc_id: str,
) -> str:
    """
    根据 doc_id 删除一个文档。

    删除内容包含：

        1. Chroma 中该文档的所有 Chunk 向量
        2. data/uploads/ 中保存的原始文件

    返回一段中文结果说明，方便直接显示在前端。
    """

    if not doc_id:

        return "❌ 没有选择要删除的文档。"

    collection = _get_collection()

    if collection is None:

        return "❌ 知识库为空，或暂时无法访问。"

    # --------------------------------------------------------
    # 1. 找出这个文档的所有 Chunk
    # --------------------------------------------------------

    try:

        data = collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )

    except Exception as exc:

        return f"❌ 读取文档失败：{exc}"

    ids = data.get("ids") or []

    if not ids:

        return "❌ 没有找到该文档，可能已经被删除。"

    metadatas = data.get("metadatas") or []

    saved_filename = ""

    for metadata in metadatas:

        if metadata and metadata.get(
            "saved_filename"
        ):

            saved_filename = metadata[
                "saved_filename"
            ]

            break

    # --------------------------------------------------------
    # 2. 删除向量
    # --------------------------------------------------------

    try:

        collection.delete(ids=ids)

    except Exception as exc:

        return f"❌ 删除向量失败：{exc}"

    # --------------------------------------------------------
    # 3. 删除原始文件
    # --------------------------------------------------------

    file_note = ""

    if saved_filename:

        saved_path = (
            UPLOAD_DIR / saved_filename
        )

        try:

            if saved_path.exists():

                Path(saved_path).unlink()

                file_note = "，并已删除本地原文件"

        except Exception:

            # 文件删除失败不影响向量删除结果，
            # 所以这里只做一个提示。
            file_note = "（本地原文件删除失败）"

    return (
        f"✅ 已删除文档，共移除 "
        f"{len(ids)} 个 Chunk{file_note}。"
    )