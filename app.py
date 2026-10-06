"""
TalentMatch AI
程序入口

运行：

    python app.py

然后浏览器访问：

    http://127.0.0.1:7860

如果 7860 端口已被占用（例如上一次的服务还没关闭），
程序会自动向后寻找一个空闲端口，
并在控制台打印真正使用的地址。
"""

import socket
import sys

from src.ui import (
    CSS,
    build_app,
    build_theme,
)


# ============================================================
# 端口选择
# ============================================================

DEFAULT_PORT = 7860

# 从 7860 开始，最多往后试多少个端口。
PORT_COUNT = 20


def find_free_port(
    start: int = DEFAULT_PORT,
    count: int = PORT_COUNT,
) -> int | None:
    """
    从 start 开始，寻找第一个可以绑定的空闲端口。

    Returns:
        int | None:
            可用端口；全部被占用时返回 None。

    说明：

        这里只是「先试着绑定一下」来探测端口，
        真正的监听还是交给 Gradio。

        注意：Windows 上不能设置 SO_REUSEADDR，
        否则会允许绑定到已被占用的端口，
        导致探测失效。
    """

    for port in range(start, start + count):

        with socket.socket(
            socket.AF_INET,
            socket.SOCK_STREAM,
        ) as sock:

            try:

                sock.bind(("127.0.0.1", port))

            except OSError:

                # 该端口已被占用，继续试下一个。
                continue

            return port

    return None


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    port = find_free_port()

    if port is None:

        raise SystemExit(
            "找不到可用端口，"
            "请关闭占用 7860 附近的程序后重试。"
        )

    # 创建 Gradio App。
    app = build_app()

    if port != DEFAULT_PORT:

        print(
            f"[提示] 端口 {DEFAULT_PORT} 已被占用，"
            f"自动改用 {port}。"
        )

    print(
        f"[启动] TalentMatch AI 地址："
        f"http://127.0.0.1:{port}"
    )

    # 启动本地 Web 服务。
    app.launch(

        # 只允许本机访问。
        server_name="127.0.0.1",

        # Web UI 端口。
        server_port=port,

        # 出错时在 Gradio 页面显示错误。
        show_error=True,

        # Gradio 6：样式与主题在 launch() 时传入。
        css=CSS,
        theme=build_theme(),

        # 双击 exe 启动时自动打开浏览器；
        # 源码运行时不自动打开，避免开发时反复弹窗。
        inbrowser=getattr(sys, "frozen", False),
    )