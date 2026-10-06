"""
测试文本 Chunk。
"""

from src.ingest import (
    get_text_splitter,
)


def test_chunking():

    splitter = get_text_splitter()

    text = (
        "这是第一段内容。\n\n"
        "这是第二段内容。\n\n"
        "这是第三段内容。"
    )

    chunks = splitter.split_text(
        text
    )

    # 至少应该产生一个 Chunk。
    assert len(chunks) >= 1

    # 每个 Chunk 都应该是字符串。
    assert all(
        isinstance(
            chunk,
            str,
        )
        for chunk in chunks
    )