"""
TalentMatch AI
向量数据库模块

负责：

1. 加载 BGE-M3
2. 将文本转换为向量
3. 连接 Chroma
4. 为 RAG 提供向量数据库

RAG 的核心流程：

文档
 ↓
Chunk
 ↓
BGE-M3
 ↓
Embedding Vector
 ↓
Chroma

用户问题
 ↓
BGE-M3
 ↓
Query Vector
 ↓
Chroma 相似度检索
 ↓
相关 Chunk
"""

from functools import lru_cache

# 注意导入顺序：
#
# src.config 必须在 FlagEmbedding 之前导入。
#
# 因为 huggingface_hub 是在被导入时读取 HF_ENDPOINT，
# 而 config 会（通过 src/__init__.py）设置好国内镜像地址。
import os

from src.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LOCAL_EMBEDDING_DIR,
)

from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from FlagEmbedding import BGEM3FlagModel, FlagModel


# ============================================================
# 模型路径解析
# ============================================================

def resolve_model_path() -> str:
    """
    决定使用哪个 Embedding 模型。

    优先级：

        1. 本地目录 models/bge-m3
           （运行过 download_model.py，推荐，完全不联网）

        2. 环境变量 / .env 里的 EMBEDDING_MODEL

    为什么要有这个函数？

        FlagEmbedding 内部逻辑是：

            if not os.path.exists(路径):
                snapshot_download(...)

        也就是说，只要传一个「存在的本地目录」，
        它就会直接加载，不会触发任何网络请求。

        这就绕开了两个坑：

            - huggingface.co 在国内无法访问
            - 镜像上 imgs/.DS_Store 返回 403 导致下载失败
    """

    # 本地目录存在，并且确实有权重文件。
    if LOCAL_EMBEDDING_DIR.is_dir():

        has_weight = (
            any(LOCAL_EMBEDDING_DIR.glob("*.bin"))
            or any(LOCAL_EMBEDDING_DIR.glob("*.safetensors"))
        )

        if has_weight:

            return str(LOCAL_EMBEDDING_DIR)

    # 退回配置里的模型名（此时会联网下载）。
    return EMBEDDING_MODEL


# ============================================================
# BGE-M3 Embedding Adapter
# ============================================================

class BgeM3Embeddings(Embeddings):
    """
    LangChain Embeddings 接口的 BGE-M3 实现。

    LangChain 需要两个方法：

        embed_documents()
            文档 Embedding

        embed_query()
            用户问题 Embedding
    """

    def __init__(self):

        model_path = resolve_model_path()

        self.model_path = model_path

        if os.path.isdir(model_path):

            print(
                f"[Embedding] 使用本地模型：{model_path}"
            )

        else:

            print(
                f"[Embedding] 正在加载模型：{model_path}"
            )

            print(
                "[Embedding] 未找到本地模型，"
                "将从下面这个源下载（约 2.3GB）："
                f"{os.environ.get('HF_ENDPOINT')}"
            )

            print(
                "[Embedding] 建议先运行 "
                "python download_model.py 预先下载。"
            )

        # 根据模型类型选择加载器：
        #
        #   bge-m3              → BGEM3FlagModel
        #                         encode() 返回 {"dense_vecs": ndarray}
        #
        #   bge-small/base-zh   → FlagModel
        #                         encode() 直接返回 ndarray
        #
        # 两者返回结构不同，所以在 _encode() 里统一处理。
        lowered = str(model_path).lower()

        self.is_m3 = "bge-m3" in lowered

        # bge-m3 支持 8192 长度；small/base 一般用 512
        self.max_length = 8192 if self.is_m3 else 512

        if self.is_m3:

            self.model = BGEM3FlagModel(
                model_path,
                use_fp16=False,
            )

        else:

            self.model = FlagModel(
                model_path,
                use_fp16=False,
            )

        print(
            f"[Embedding] 模型加载完成"
            f"（{'BGE-M3' if self.is_m3 else 'BGE 中文模型'}）。"
        )


    # ========================================================
    # 文档 Embedding
    # ========================================================

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        将多个文本转换为向量。

        用于：
            JD Chunk
            Resume Chunk

        Args:
            texts:
                文本列表。

        Returns:
            向量列表。
        """

        if not texts:
            return []

        return self._encode(
            texts,
            batch_size=8,
        )


    # ========================================================
    # Query Embedding
    # ========================================================

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """
        将用户查询转换成向量。

        例如：

        「有没有会 RAG 的 AI 产品经理？」

        ↓

        BGE-M3

        ↓

        Query Vector
        """

        return self._encode(
            [text],
            batch_size=1,
        )[0]


    # ========================================================
    # 统一编码入口
    # ========================================================

    def _encode(
        self,
        texts: list[str],
        batch_size: int,
    ) -> list[list[float]]:
        """
        把文本编码成向量列表。

        之所以要单独抽出来，是因为两种模型的返回结构不一样：

            BGEM3FlagModel.encode()  →  {"dense_vecs": ndarray}
            FlagModel.encode()       →  ndarray

        这里统一转换成 Python 的 list[list[float]]。
        """

        result = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=self.max_length,
        )

        # BGEM3 返回字典，取 dense_vecs
        if isinstance(result, dict):

            result = result["dense_vecs"]

        # numpy array → list
        if hasattr(result, "tolist"):

            return result.tolist()

        return [list(vector) for vector in result]


# ============================================================
# Embedding 单例
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings() -> BgeM3Embeddings:
    """
    获取 Embedding 模型。

    使用缓存避免每次查询都重新加载 Embedding 模型。
    """

    return BgeM3Embeddings()


# ============================================================
# Chroma
# ============================================================

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """
    获取 Chroma 向量数据库。

    数据持久化到：

        data/chroma/

    因此关闭程序后，
    已经建立的向量不会丢失。
    """

    print(
        "[Chroma] 正在连接本地向量数据库..."
    )

    store = Chroma(
        collection_name=COLLECTION_NAME,

        # 使用 BGE-M3。
        embedding_function=get_embeddings(),

        # 本地持久化目录。
        persist_directory=str(
            CHROMA_DIR
        ),

        # cosine distance
        collection_metadata={
            "hnsw:space": "cosine"
        },
    )

    print(
        "[Chroma] 连接完成。"
    )

    return store