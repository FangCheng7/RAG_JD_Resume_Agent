"""
TalentMatch AI
BGE-M3 模型下载脚本

用法：

    python download_model.py

为什么要单独提供这个脚本？

    1. 模型约 2.3GB，第一次使用时自动下载，
       界面会一直显示 processing，体验很差。

    2. huggingface.co 在国内通常无法直连，
       所以这里默认使用国内镜像 hf-mirror.com。

    3. 只下载推理真正需要的文件，
       跳过 onnx / pytorch_model.bin / imgs 等无用内容，
       能省下一半以上的体积。

下载完成后，模型会保存在：

    models/bge-m3/

之后 src/vectorstore.py 会自动优先使用这个本地目录，
不再联网。
"""

import os
import sys
from pathlib import Path


# ============================================================
# 配置
# ============================================================

REPO_ID = "BAAI/bge-m3"

# 复用 config 里的 BASE_DIR：
# 源码运行时是项目根目录，
# 打包成 exe 后是 exe 所在目录，
# 两种情况下模型都放在同一个相对位置。
from src.config import BASE_DIR

TARGET_DIR = BASE_DIR / "models" / "bge-m3"

# 默认使用国内镜像；
# 如需使用官方源，可设置环境变量 HF_ENDPOINT 覆盖。
os.environ.setdefault(
    "HF_ENDPOINT",
    "https://hf-mirror.com",
)

# 关闭 HuggingFace 的 Xet 传输。
#
# 大文件默认会走 Xet 存储（cas-server.xethub.hf.co），
# 这个域名镜像不代理，会直接 401 Unauthorized。
# 关掉之后会回退到普通 HTTP 下载，镜像就能正常提供文件。
os.environ.setdefault(
    "HF_HUB_DISABLE_XET",
    "1",
)

# 只下载推理需要的文件。
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
    "*.bin",
    "*.pt",
]

# 明确排除的内容：
#   imgs/  仓库里的图片与 .DS_Store（镜像上会返回 403，导致整个下载失败）
#   onnx/  ONNX 版本，体积很大且这里用不到
#
# 注意：BAAI/bge-m3 只提供 pytorch_model.bin（约 2.2GB），
#       没有 model.safetensors，所以 *.bin 必须保留。
IGNORE_PATTERNS = [
    "imgs/*",
    "onnx/*",
    "*.onnx",
    "*.h5",
    "*.msgpack",
    "*.ot",
]


# ============================================================
# 主流程
# ============================================================

def main() -> int:

    from huggingface_hub import snapshot_download

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"[下载] 仓库：{REPO_ID}")
    print(f"[下载] 来源：{os.environ['HF_ENDPOINT']}")
    print(f"[下载] 目录：{TARGET_DIR}")
    print("[下载] 约 2.3GB，请耐心等待……")
    print()

    try:

        snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(TARGET_DIR),
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
            max_workers=4,
        )

    except Exception as exc:

        print()
        print(f"[失败] {type(exc).__name__}: {exc}")
        print()
        print("可以尝试：")
        print("  1. 检查网络是否能访问 hf-mirror.com")
        print("  2. 设置 HF_ENDPOINT 指向其他可用镜像")
        print("  3. 重新运行本脚本（已下载的部分会跳过）")

        return 1

    # 校验关键文件是否齐全。
    weights = (
        list(TARGET_DIR.glob("*.safetensors"))
        + list(TARGET_DIR.glob("*.bin"))
    )

    if not weights:

        print("[失败] 没有找到模型权重文件（*.bin / *.safetensors）。")

        return 1

    total = sum(
        f.stat().st_size
        for f in TARGET_DIR.rglob("*")
        if f.is_file()
    )

    print()
    print(f"[完成] 权重文件：{weights[0].name}")
    print(f"[完成] 总大小：{total / 1024 / 1024 / 1024:.2f} GB")
    print("[完成] 现在可以运行 python app.py")

    return 0


if __name__ == "__main__":

    sys.exit(main())