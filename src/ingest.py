"""
TalentMatch AI
文档导入模块

负责：

PDF / DOCX
    ↓
文本解析（pypdf / python-docx）
    ↓
纯文本
    ↓
Chunk
    ↓
BGE-M3
    ↓
Chroma

当前版本：

支持：
    PDF（文字版 / 电子版）
    DOCX
    直接粘贴文本

暂不支持：
    扫描件 / 图片型 PDF 的 OCR

说明：
    图片型 PDF 没有文字层，无法直接提取。
    遇到这种情况会给出明确提示，
    引导用户改用「直接输入文本」。
"""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    UPLOAD_DIR,
)

from src.vectorstore import (
    get_vectorstore,
)


# ============================================================
# 支持的文件
# ============================================================

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
}


# ============================================================
# 文档解析
# ============================================================
#
# 这里用轻量的纯 Python 库直接解析：
#
#     PDF  → pypdf
#     DOCX → python-docx
#
# 为什么不再用 Unstructured？
#
#   1. Unstructured 的 PDF 解析依赖 unstructured_inference，
#      这个包经常装不上；缺了它会直接抛
#      ModuleNotFoundError，表现就是「导入 PDF 没反应」。
#
#   2. 它会连带引入 spacy / nltk 等一大堆依赖（几百 MB），
#      但对「提取纯文本」这个需求没有任何帮助。
#
#   3. pypdf + python-docx 更小、更快、更稳定。
#
# 注意：
#   本项目只处理「电子版」PDF。
#   扫描件 / 图片型 PDF 没有文字层，提取不到内容，
#   这时会给出明确提示，引导用户改用文本录入。
# ============================================================


def _extract_pdf_text(path: Path) -> str:
    """
    用 pypdf 提取电子版 PDF 的文字。

    扫描件（整页都是图片）提取结果是空字符串，
    由调用方负责提示用户。
    """

    from pypdf import PdfReader

    reader = PdfReader(str(path))

    parts = []

    for page in reader.pages:

        try:

            parts.append(
                page.extract_text() or ""
            )

        except Exception:

            # 单页解析失败不影响其他页。
            continue

    return "\n".join(parts).strip()


def _iter_docx_blocks(document):
    """
    按「文档真实顺序」遍历段落与表格。

    python-docx 的 paragraphs 和 tables 是两个独立列表，
    直接遍历会丢失顺序，
    而 JD / 简历经常用表格排版，所以需要按 body 顺序走。
    """

    from docx.oxml.table import CT_Tbl
    from docx.oxml.text.paragraph import CT_P
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in document.element.body.iterchildren():

        if isinstance(child, CT_P):

            yield Paragraph(child, document)

        elif isinstance(child, CT_Tbl):

            yield Table(child, document)


def _extract_docx_text(path: Path) -> str:
    """
    用 python-docx 提取 DOCX 文字，包含表格内容。
    """

    from docx import Document

    document = Document(str(path))

    parts = []

    for block in _iter_docx_blocks(document):

        # ------------------------------------------------
        # 段落
        # ------------------------------------------------

        if hasattr(block, "text"):

            text = block.text.strip()

            if text:

                parts.append(text)

            continue

        # ------------------------------------------------
        # 表格：按行拼接
        # ------------------------------------------------

        for row in block.rows:

            cells = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if cells:

                parts.append(" | ".join(cells))

    return "\n".join(parts).strip()


def extract_document_text(
    path: Path,
) -> str:
    """
    解析 PDF / DOCX，得到纯文本。

    Args:
        path:
            文件路径。

    Returns:
        str:
            提取出来的纯文本。

    Raises:
        ValueError:
            文件格式不支持，或提取不到任何文字。
    """

    suffix = path.suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:

        raise ValueError(
            f"不支持的文件类型：{suffix}"
        )

    print(
        f"[Parser] 正在解析："
        f"{path.name}"
    )

    # --------------------------------------------------------
    # 按类型解析
    # --------------------------------------------------------

    if suffix == ".pdf":

        text = _extract_pdf_text(path)

    else:

        text = _extract_docx_text(path)

    text = text.strip()

    # --------------------------------------------------------
    # 文本太少：大概率是扫描件 / 图片型 PDF
    # --------------------------------------------------------

    if len(text) < 40:

        if suffix == ".pdf":

            raise ValueError(
                "这是扫描件 / 图片型 PDF，"
                "没有文字层，提取不到内容。"
                "请改用「方式二 · 直接输入」粘贴文字。"
            )

        raise ValueError(
            "文档里没有提取到有效文字，"
            "请检查文件内容是否为空。"
        )

    print(
        f"[Parser] 解析完成："
        f"{len(text)} 字符"
    )

    return text

# ============================================================
# Chunk Splitter
# ============================================================

def get_text_splitter():
    """
    创建文本切分器。

    Chunk 的意义：

    原始 JD
        ↓
    很长的一篇文本

    切成：

    Chunk 1
    Chunk 2
    Chunk 3
    ...

    然后每个 Chunk 单独进行 Embedding。
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,

        # 优先按照语义结构切分。
        separators=[
            "\n\n",
            "\n",
            "。",
            "；",
            "，",
            " ",
            "",
        ],
    )


# ============================================================
# 单文件导入
# ============================================================

def ingest_one_file(
    file_path: str | Path,
    doc_type: str,
) -> str:
    """
    导入一个 JD 或简历。

    Args:
        file_path:
            文件路径。

        doc_type:
            jd / resume

    Returns:
        导入结果。
    """

    if doc_type not in {
        "jd",
        "resume",
    }:

        raise ValueError(
            "doc_type 必须是 jd 或 resume"
        )

    source = Path(file_path)

    if not source.exists():

        raise FileNotFoundError(
            f"文件不存在：{source}"
        )

    # --------------------------------------------------------
    # Step 1：解析文档
    # --------------------------------------------------------

    text = extract_document_text(
        source
    )

    # --------------------------------------------------------
    # Step 2：生成唯一 document_id
    # --------------------------------------------------------

    doc_id = str(
        uuid4()
    )

    # --------------------------------------------------------
    # Step 3：保存原始文件
    # --------------------------------------------------------

    saved_name = (
        f"{doc_id}_{source.name}"
    )

    saved_path = (
        UPLOAD_DIR / saved_name
    )

    saved_path.write_bytes(
        source.read_bytes()
    )

    # --------------------------------------------------------
    # Step 4：切分 + 写入 Chroma
    # --------------------------------------------------------
    #
    # 具体逻辑抽取到 _store_chunks()，
    # 这样「上传文件」和「直接粘贴文本」可以共用同一套入库流程。
    # --------------------------------------------------------

    chunk_count = _store_chunks(
        text=text,
        doc_id=doc_id,
        filename=source.name,
        doc_type=doc_type,
        saved_filename=saved_name,
    )

    return (
        f"成功：{source.name}\n"
        f"document_id：{doc_id}\n"
        f"类型：{doc_type}\n"
        f"chunks：{chunk_count}"
    )


# ============================================================
# Chunk 写入（文件 / 直接文本 共用）
# ============================================================

def _store_chunks(
    text: str,
    doc_id: str,
    filename: str,
    doc_type: str,
    saved_filename: str,
) -> int:
    """
    把一段文本切分、写入 Chroma。

    Args:
        text:
            已经解析出来的纯文本。

        doc_id:
            文档唯一 ID。

        filename:
            展示用的文件名。

        doc_type:
            jd / resume

        saved_filename:
            本地保存的文件名。

    Returns:
        int:
            写入的 chunk 数量。
    """

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    splitter = get_text_splitter()

    chunks = splitter.split_text(text)

    if not chunks:

        raise ValueError(
            "文本切分后没有产生任何 Chunk。"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadatas = []
    ids = []

    for index, chunk in enumerate(chunks):

        # 每个 Chunk 都有自己的 ID。
        ids.append(f"{doc_id}_{index}")

        # Metadata 非常重要。
        #
        # 以后可以通过：
        #
        #     doc_type=resume
        #
        # 只搜索简历。
        metadatas.append(
            {
                "doc_type": doc_type,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": index,
                "saved_filename": saved_filename,
            }
        )

    # --------------------------------------------------------
    # 写入 Chroma
    # --------------------------------------------------------

    store = get_vectorstore()

    store.add_texts(
        texts=chunks,
        metadatas=metadatas,
        ids=ids,
    )

    print(
        f"[Chroma] {filename} → {len(chunks)} chunks"
    )

    return len(chunks)


def _safe_filename(
    title: str,
    fallback: str,
) -> str:
    """
    把用户输入的标题转成安全的文件名。

    会去掉 Windows 文件名里的非法字符。
    """

    title = (title or "").strip()

    if not title:

        return fallback

    for char in '\\/:*?"<>|':

        title = title.replace(char, " ")

    # 多个空格合并成一个。
    title = " ".join(title.split())

    # 限制长度，避免文件名过长。
    return title[:60] or fallback


# ============================================================
# 直接导入文本（不需要 PDF / DOCX）
# ============================================================

def ingest_text(
    text: str,
    doc_type: str,
    title: str = "",
) -> str:
    """
    直接导入一段文本，不经过 PDF / DOCX。

    使用场景：

        用户手里只有一段 JD 文字时，
        可以直接粘贴进来，不用先转成 PDF 再上传。

    Args:
        text:
            JD / 简历的正文内容。

        doc_type:
            jd / resume

        title:
            可选标题，会作为文件名与展示名。

    Returns:
        str:
            导入结果说明。
    """

    if doc_type not in {
        "jd",
        "resume",
    }:

        raise ValueError(
            "doc_type 必须是 jd 或 resume"
        )

    text = (text or "").strip()

    # 太短的内容基本不可能是有效的 JD / 简历，
    # 提前拦掉，避免污染知识库。
    if len(text) < 20:

        raise ValueError(
            "文本内容太短（至少需要 20 个字符）。"
        )

    kind = "JD" if doc_type == "jd" else "简历"

    fallback = (
        f"{kind}-"
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    display_name = (
        _safe_filename(title, fallback)
        + ".txt"
    )

    doc_id = str(uuid4())

    saved_name = f"{doc_id}_{display_name}"

    # --------------------------------------------------------
    # 保存一份原文
    # --------------------------------------------------------
    #
    # 和上传文件保持一致的目录结构，
    # 这样删除文档时也能一起清理掉。
    # --------------------------------------------------------

    saved_path = UPLOAD_DIR / saved_name

    saved_path.write_text(
        text,
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 切分 + 入库
    # --------------------------------------------------------

    chunk_count = _store_chunks(
        text=text,
        doc_id=doc_id,
        filename=display_name,
        doc_type=doc_type,
        saved_filename=saved_name,
    )

    return (
        f"成功：{display_name}\n"
        f"document_id：{doc_id}\n"
        f"类型：{doc_type}\n"
        f"chunks：{chunk_count}"
    )

# ============================================================
# 批量导入
# ============================================================

def ingest_files(
    files,
    doc_type: str,
) -> dict:
    """
    批量导入文件。

    一个文件失败不会影响其他文件。

    Args:
        files:
            文件路径列表。

        doc_type:
            jd / resume

    Returns:
        dict:
            {
                "ok": 成功数量,
                "failed": 失败数量,
                "summary": 一句话摘要（前端弹窗用）,
                "messages": [每个文件的详细结果],
            }
    """

    if not files:

        return {
            "ok": 0,
            "failed": 0,
            "summary": "请先选择文件。",
            "messages": [],
        }

    ok_names: list[str] = []
    failed_names: list[str] = []
    messages: list[str] = []

    first_error = ""

    for file_item in files:

        name = Path(str(file_item)).name

        try:

            result = ingest_one_file(
                str(file_item),
                doc_type,
            )

            ok_names.append(name)

            messages.append(f"✅ {name}\n{result}")

        except Exception as exc:

            failed_names.append(name)

            if not first_error:

                # 只取第一行，方便放进弹窗。
                first_error = str(exc).splitlines()[0]

            messages.append(f"❌ {name}\n{exc}")

            print(f"[Ingest] 导入失败 {name}：{exc}")

    # --------------------------------------------------------
    # 生成一句话摘要
    # --------------------------------------------------------

    if ok_names and not failed_names:

        summary = f"成功导入 {len(ok_names)} 个文件"

    elif failed_names and not ok_names:

        if len(failed_names) == 1:

            summary = f"{failed_names[0]}：{first_error}"

        else:

            summary = (
                f"{len(failed_names)} 个文件导入失败："
                f"{first_error}"
            )

    else:

        summary = (
            f"成功 {len(ok_names)} 个，"
            f"失败 {len(failed_names)} 个：{first_error}"
        )

    return {
        "ok": len(ok_names),
        "failed": len(failed_names),
        "summary": summary,
        "messages": messages,
    }
