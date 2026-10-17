# JD-简历智能匹配 Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows-0078D6.svg)]()
[![Release](https://img.shields.io/github/v/release/FangCheng7/RAG_JD_Resume_Agent?label=Release&color=brightgreen)](https://github.com/FangCheng7/RAG_JD_Resume_Agent/releases/latest)

一个基于 **RAG + LangGraph** 的 JD / 简历智能匹配 AI Agent。

上传岗位 JD 与候选人简历后，可以直接用自然语言完成检索、解析、匹配、
候选人排序与对比。

---

## ⬇️ 不想装 Python？直接下载 exe

到 **[Releases](https://github.com/FangCheng7/RAG_JD_Resume_Agent/releases/latest)** 下载 `TalentMatchAI.zip`，
解压后双击 `TalentMatchAI.exe` 即可使用（会自动打开浏览器）。

> - **向量模型已内置在压缩包里，下载即用，不需要联网下载模型。**
> - 详细说明见压缩包内的 `启动说明.txt`。
> - 需要 VC++ 运行库（Win10/11 一般自带，老系统装
>   [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) 即可）。

---

## 技术栈

| 层次 | 技术 |
| --- | --- |
| 语言 | Python 3.11+ |
| Agent | LangGraph / LangChain |
| LLM | DeepSeek（OpenAI 兼容接口） |
| Embedding | BGE-small-zh-v1.5（本地运行，随包分发） |
| 向量库 | Chroma |
| 文档解析 | pypdf（PDF）+ python-docx（DOCX） |
| 前端 | Gradio |

---

## 核心能力

### 1. JD 知识库

批量上传 JD（PDF / DOCX），或直接粘贴文本，自动完成：

```
文档 → 文本解析 → Chunk → BGE-small-zh → Chroma
```

### 2. 简历知识库

批量上传候选人简历，建立独立的向量索引。

### 3. RAG 检索

Agent 可根据问题自主检索：

- 搜索 JD
- 搜索简历
- 获取完整文档

### 4. 结构化解析

- **JD**：岗位名称、部门、硬性要求、技能、工作职责、学历、工作年限
- **简历**：姓名、工作年限、学历、技能、工作经历、项目经历

### 5. JD-简历匹配

输出 0-100 匹配分、匹配点、不足与综合评价，并对候选人排序。

> 评分仅作为辅助决策依据，不使用年龄、性别、民族等受保护属性，
> 最终录用决定必须由人工完成。

---

## 界面

前端是一个单页的招聘工作台，包含 6 个页面：

| 页面 | 说明 |
| --- | --- |
| 概览 | 知识库统计、RAG 链路、快速入口 |
| JD 管理 | 上传 JD、查看已入库文档、删除文档 |
| 简历库 | 上传简历、查看已入库文档、删除文档 |
| AI Agent | 多轮对话式招聘分析 |
| 数据管理 | 全局文档总览与清理 |
| 系统设置 | 配置 DeepSeek API、测试连接 |

---

## 快速开始

### 1. 安装依赖

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

> 首次运行需要 Embedding 模型（约 180MB）。
> 国内网络无法直连 huggingface.co，
> 请先执行一次 `python download_model.py` 下载（默认走 hf-mirror.com 镜像）。

### 2. 配置 API Key

复制 `.env.example` 为 `.env`，填入自己的 DeepSeek API Key：

```env
DEEPSEEK_API_KEY=sk-your-own-key
```

也可以直接启动后在 **系统设置** 页面填写，配置会保存到
`data/config.json`（该文件已被 `.gitignore` 忽略）。

### 3. 下载向量模型（仅源码运行需要）

```bash
python download_model.py
```

约 180MB，默认使用国内镜像 `hf-mirror.com`。
> （打包成 exe 时会自动带上模型，最终用户无需下载。）
下载后再启动，就不会出现「页面一直 processing」。

### 4. 打包成 exe（可选，给别人用）

如果想让别人**不用装 Python** 就能用，项目已内置 PyInstaller 配置：

```bat
:: Windows 下直接双击
build_exe.bat

:: 或者手动执行
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean TalentMatchAI.spec
```

打包产物：

```
dist/TalentMatchAI/TalentMatchAI.exe
```

把整个 `TalentMatchAI` 文件夹压缩后发给别人，对方解压后双击 exe 即可使用（会自动打开浏览器）。

> **模型已随包分发**（打包脚本会自动把 `models/` 复制进产物），
> 最终用户下载解压后直接双击即可使用，不需要任何额外下载。

### 5. 启动

```bash
python app.py
```

浏览器访问：<http://127.0.0.1:7860>

---

## 项目结构

```
rag_jd_resume_agent/
├── app.py                 # 程序入口
├── requirements.txt
├── data/                  # 本地数据（已忽略）
│   ├── chroma/            #   向量库
│   └── uploads/           #   上传的原始文件
└── src/
    ├── agent.py           # LangGraph Agent 与聊天接口
    ├── config.py          # 配置管理
    ├── ingest.py          # 文档解析（pypdf / python-docx）/ 切分 / 入库
    ├── llm.py             # DeepSeek LLM
    ├── services.py        # 业务服务层（统计 / 列表 / 删除）
    ├── tools.py           # Agent 工具
    ├── ui.py              # Gradio 前端
    └── vectorstore.py     # BGE Embedding + Chroma
```

---

## 安全提醒

- **不要把 `.env` 或 `data/config.json` 提交到 Git**（已在 `.gitignore` 中忽略）。
- 如果 API Key 曾经出现在截图、聊天记录或代码中，请立即到
  DeepSeek 控制台**吊销并重新生成**。
- 本项目默认只监听 `127.0.0.1`，不对外网开放。

---

## 测试

```bash
pytest
```