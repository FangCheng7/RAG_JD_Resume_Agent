"""
测试项目核心模块是否能够正常导入。

这个测试主要用于发现：

- ImportError
- ModuleNotFoundError
- 语法错误
- 文件之间的依赖问题
"""


def test_imports():

    # 配置
    import src.config

    # LLM
    import src.llm

    # Vector DB
    import src.vectorstore

    # 文档处理
    import src.ingest

    # Tools
    import src.tools

    # Agent
    import src.agent

    # Services
    import src.services

    # UI
    import src.ui

    # 主程序
    import app