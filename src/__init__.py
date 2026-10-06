"""
TalentMatch AI
包初始化

这里负责设置 HuggingFace 的下载源。

为什么放在 __init__.py？

    BGE-M3 需要从 HuggingFace 下载（约 2.3GB）。
    国内网络通常无法直连 huggingface.co，
    表现就是「页面一直 processing」——
    实际是卡在模型下载的 TCP 握手上。

    huggingface_hub 是在「被导入时」读取 HF_ENDPOINT 环境变量，
    而 src/__init__.py 一定比其他 src 子模块先执行，
    所以放在这里最可靠。

如何改回官方源？

    在项目根目录的 .env 里写：

        HF_ENDPOINT=https://huggingface.co

    .env 里的值优先级更高。
"""

import os

# 先读取 .env，让用户可以用 .env 覆盖下载源。
try:

    from dotenv import load_dotenv

    load_dotenv()

except Exception:

    pass

# 没有显式配置时，默认使用国内镜像。
os.environ.setdefault(
    "HF_ENDPOINT",
    "https://hf-mirror.com",
)

# 关闭 Xet 传输。
#
# 大文件默认走 Xet（cas-server.xethub.hf.co），
# 该域名不被镜像代理，会 401。
# 关闭后回退普通 HTTP 下载。
os.environ.setdefault(
    "HF_HUB_DISABLE_XET",
    "1",
)