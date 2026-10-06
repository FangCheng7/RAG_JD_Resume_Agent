# ============================================================
# src/ui.py
# TalentMatch AI
# 前端 UI（重新制作版）
#
# 页面包括：
#   1. 概览        —— 知识库统计 / 快速入口 / RAG 流程
#   2. JD 管理     —— 上传 + 文档列表 + 删除
#   3. 简历库      —— 上传 + 文档列表 + 删除
#   4. AI Agent    —— 多轮对话式招聘分析
#   5. 数据管理    —— 全局文档总览与清理
#   6. 系统设置    —— DeepSeek API 配置与连接测试
#
# 设计说明：
#   这是一个「暗色 + 玻璃拟态」的 AI 工作台界面，
#   通过全局 CSS 覆盖 Gradio 默认样式实现。
#
# 本文件只负责 UI 与页面交互，
# 业务逻辑全部位于：
#   src/agent.py / src/config.py / src/ingest.py
#   src/llm.py  / src/services.py
# ============================================================


# ============================================================
# 一、导入依赖
# ============================================================

import gradio as gr

from src.agent import (
    chat,
    clear_agent_cache,
)

from src.config import (
    get_api_key,
    get_base_url,
    get_model,
    save_user_config,
)

from src.ingest import (
    ingest_files,
    ingest_text,
)

from src.llm import clear_llm_cache

from src.services import (
    delete_document,
    get_dashboard_stats,
    get_documents,
)


# ============================================================
# 二、全局配置
# ============================================================

APP_TITLE = "TalentMatch AI"
APP_SLOGAN = "企业 JD · 简历智能匹配工作台 -- By Fang Cheng" 


# 侧边栏导航
# (显示文本, 图标)
NAV_ITEMS = [
    ("概览", "◎"),
    ("JD 管理", "▤"),
    ("简历库", "❏"),
    ("AI Agent", "✦"),
    ("数据管理", "◈"),
    ("系统设置", "⚙"),
]

# 页面容器 / 导航按钮的索引
PAGE_OVERVIEW = 0
PAGE_JD = 1
PAGE_RESUME = 2
PAGE_AGENT = 3
PAGE_DATA = 4
PAGE_SETTINGS = 5


# ============================================================
# 三、全局 CSS（设计系统）
# ============================================================

CSS = """

/* ============================================================
   0. 强制亮色
   ============================================================ */
/* Gradio 内置组件（下拉、上传、聊天气泡等）的颜色都来自
   这些 CSS 变量。这里把「浅色」和「深色」两套变量都覆盖成
   亮色值，这样即使系统是深色模式，界面也保持亮色。 */

.gradio-container,
.gradio-container.dark,
.dark,
:root {
    --body-background-fill: #eaeff7 !important;
    --body-background-fill-dark: #eaeff7 !important;

    --background-fill-primary: #ffffff !important;
    --background-fill-primary-dark: #ffffff !important;
    --background-fill-secondary: #eaeff7 !important;
    --background-fill-secondary-dark: #eaeff7 !important;

    --block-background-fill: #ffffff !important;
    --block-background-fill-dark: #ffffff !important;
    --block-border-color: #e5e9f2 !important;
    --block-border-color-dark: #e5e9f2 !important;

    --border-color-primary: #e5e9f2 !important;
    --border-color-primary-dark: #e5e9f2 !important;

    --body-text-color: #1e2a3a !important;
    --body-text-color-dark: #1e2a3a !important;
    --body-text-color-subdued: #5b6b82 !important;
    --body-text-color-subdued-dark: #5b6b82 !important;

    --block-label-text-color: #5b6b82 !important;
    --block-label-text-color-dark: #5b6b82 !important;
    --block-title-text-color: #1e2a3a !important;
    --block-title-text-color-dark: #1e2a3a !important;

    --input-background-fill: #ffffff !important;
    --input-background-fill-dark: #ffffff !important;
    --input-border-color: #cfd7e6 !important;
    --input-border-color-dark: #cfd7e6 !important;
    --input-placeholder-color: #9aa6b8 !important;
    --input-placeholder-color-dark: #9aa6b8 !important;

    --panel-background-fill: #ffffff !important;
    --panel-background-fill-dark: #ffffff !important;

    --table-even-background-fill: #ffffff !important;
    --table-even-background-fill-dark: #ffffff !important;
    --table-odd-background-fill: #f8fafc !important;
    --table-odd-background-fill-dark: #f8fafc !important;
    --table-border-color: #e5e9f2 !important;
    --table-border-color-dark: #e5e9f2 !important;

    --color-accent: #4f46e5 !important;
    --color-accent-dark: #4f46e5 !important;
    --color-accent-soft: #eef2ff !important;
    --color-accent-soft-dark: #eef2ff !important;

    --checkbox-background-color: #ffffff !important;
    --checkbox-background-color-dark: #ffffff !important;
    --checkbox-label-background-fill: #ffffff !important;
    --checkbox-label-background-fill-dark: #ffffff !important;
}


/* ============================================================
   1. 基础变量
   ============================================================ */

:root {
    --bg: #eaeff7;
    --panel: #ffffff;
    --panel-soft: #f8fafc;

    --line: #e5e9f2;
    --line-strong: #cfd7e6;

    --text: #1e2a3a;
    --text-dim: #5b6b82;
    --text-mute: #8b98ac;

    --indigo: #4f46e5;
    --violet: #7c3aed;
    --emerald: #059669;
    --amber: #d97706;
    --rose: #e11d48;

    --radius: 14px;

    --shadow-sm: 0 1px 2px rgba(15, 23, 42, 0.06);
    --shadow-md: 0 4px 16px rgba(15, 23, 42, 0.08);
    --shadow-lg: 0 12px 32px rgba(79, 70, 229, 0.16);
}


/* ============================================================
   2. 全局
   ============================================================ */

html, body {
    margin: 0;
    padding: 0;

    background: var(--bg);
    color: var(--text);

    font-family:
        "Inter",
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        "PingFang SC",
        "Microsoft YaHei",
        sans-serif;
}

body::before {
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(
            900px 460px at 6% -10%,
            rgba(99, 102, 241, 0.10),
            transparent 62%
        ),
        radial-gradient(
            760px 420px at 100% 0%,
            rgba(139, 92, 246, 0.08),
            transparent 58%
        );
    z-index: 0;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;

    background: transparent !important;

    position: relative;
    z-index: 1;
}

.gradio-container .main,
.gradio-container .app,
.gradio-container .wrap {
    background: transparent !important;
}

.gradio-container > .main {
    padding: 0 !important;
}

/* Gradio 默认给内容区设了 max-width: 1280px，
   大屏下右侧会空出一大块，
   看起来就像「页面没占满 / 偏了」。
   这里放开限制，让内容铺满整个宽度。 */
.main.fillable,
.gradio-container .main.fillable,
.gradio-container > .main,
.gradio-container > .main.fillable,
.contain,
.wrap {
    max-width: 100% !important;
    width: 100% !important;
}


/* ============================================================
   3. 布局
   ============================================================ */

.app-row {
    gap: 0 !important;
    align-items: stretch !important;

    min-height: 100vh;

    flex-wrap: nowrap !important;

    /* 侧边栏是 fixed（脱离文档流），
       所以由父容器来预留它的宽度。
       这样右侧内容刚好填满剩余空间，不会溢出到屏幕外。 */
    padding-left: 250px !important;

    box-sizing: border-box !important;
}

.sidebar {
    flex: 0 0 250px !important;
    width: 250px !important;
    max-width: 250px !important;
    min-width: 250px !important;

    /* 固定侧边栏：
       页面往下滚时它保持不动，导航一直可见。

       这里必须用 fixed 而不是 sticky ——
       Gradio 的祖先容器带 overflow，
       会让 sticky 失效（元素照样被滚走）。

       因为 fixed 会把元素移出文档流，
       所以右侧主区域要补一个左边距。 */
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    bottom: 0 !important;

    height: 100vh;
    max-height: 100vh;
    overflow-y: auto;

    z-index: 20;

    padding: 22px 14px !important;

    background: #ffffff !important;
    border-right: 1px solid var(--line) !important;

    gap: 2px !important;
}

.main {
    flex: 1 1 auto !important;
    min-width: 0 !important;

    min-height: 100vh;

    /* 左边距交给 .app-row 的 padding 处理，
       这里保持 0，避免宽度算重导致溢出。 */
    margin-left: 0 !important;

    padding: 30px 38px 60px 38px !important;

    box-sizing: border-box !important;

    background: transparent !important;
}


/* ============================================================
   4. 品牌 Logo
   ============================================================ */

.brand {
    display: flex;
    align-items: center;
    gap: 11px;

    padding: 4px 8px 20px 8px;
}

.brand-logo {
    width: 38px;
    height: 38px;
    flex: 0 0 38px;

    border-radius: 11px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 19px;
    font-weight: 800;
    color: #ffffff;

    background: linear-gradient(135deg, var(--indigo), var(--violet));

    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.30);
}

.brand-title {
    font-size: 15px;
    font-weight: 750;
    letter-spacing: -0.2px;
    color: #16203a;
    line-height: 1.2;
}

.brand-sub {
    margin-top: 3px;
    font-size: 10px;
    color: var(--text-mute);
    letter-spacing: 0.2px;
}


/* ============================================================
   5. 侧边导航
   ============================================================ */

.nav-title {
    margin: 16px 8px 8px 8px;

    font-size: 9px;
    font-weight: 750;
    letter-spacing: 1.5px;

    color: #a3aec2;
    text-transform: uppercase;
}

.nav-btn {
    width: 100% !important;

    justify-content: flex-start !important;
    text-align: left !important;

    background: transparent !important;
    border: 1px solid transparent !important;

    color: #5b6b82 !important;

    margin: 2px 0 !important;
    padding: 10px 12px !important;

    min-height: 40px !important;

    border-radius: 10px !important;

    font-size: 13px !important;
    font-weight: 600 !important;

    box-shadow: none !important;

    transition: all 0.16s ease;
}

.nav-btn:hover {
    background: #f1f4fa !important;
    border-color: var(--line) !important;
    color: #1e2a3a !important;
}

.nav-active {
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    box-shadow: none !important;
}

.side-status {
    margin-top: 26px;
    padding: 13px 14px;

    border: 1px solid var(--line);
    border-radius: 12px;

    background: var(--panel-soft);
}

.side-status-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: #a3aec2;
}

.side-status-main {
    margin-top: 8px;
    font-size: 12px;
    font-weight: 600;
    color: var(--emerald);
}

.dot {
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-right: 6px;

    border-radius: 50%;
    background: var(--emerald);

    box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.14);
}

.side-status-sub {
    margin-top: 5px;
    font-size: 10px;
    color: var(--text-mute);
    line-height: 1.6;
}


/* ============================================================
   6. 页面标题
   ============================================================ */

.page-head {
    margin-bottom: 22px;
}

.page-eyebrow {
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 2px;
    text-transform: uppercase;

    background: linear-gradient(90deg, #4f46e5, #9333ea);
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
}

.page-title {
    margin-top: 8px;

    font-size: 27px;
    font-weight: 760;
    letter-spacing: -0.6px;

    color: #16203a;
}

.page-desc {
    margin-top: 8px;
    font-size: 13px;
    color: var(--text-dim);
    line-height: 1.6;
}


/* ============================================================
   7. 卡片 / 面板
   ============================================================ */

.stat-card {
    position: relative;
    overflow: hidden;

    padding: 18px 20px;

    min-height: 118px;

    border: 1px solid var(--line);
    border-radius: var(--radius);

    background: var(--panel);

    box-shadow: var(--shadow-sm);

    transition: transform 0.18s ease, box-shadow 0.18s ease;
}

.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.stat-card::after {
    content: "";
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
}

.tone-indigo::after { background: linear-gradient(180deg, #6366f1, #818cf8); }
.tone-violet::after { background: linear-gradient(180deg, #8b5cf6, #c084fc); }
.tone-emerald::after { background: linear-gradient(180deg, #10b981, #34d399); }
.tone-amber::after { background: linear-gradient(180deg, #f59e0b, #fbbf24); }

.stat-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    color: var(--text-mute);
    text-transform: uppercase;
}

.stat-value {
    margin-top: 12px;

    font-size: 32px;
    font-weight: 780;
    letter-spacing: -1px;

    color: #0f1a30;
    line-height: 1;
}

.stat-sub {
    margin-top: 9px;
    font-size: 11px;
    color: var(--text-dim);
}

.panel {
    padding: 20px 22px !important;

    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;

    background: var(--panel) !important;

    box-shadow: var(--shadow-sm);

    overflow: visible !important;
}

.section-title {
    margin: 26px 0 12px 0;

    font-size: 14px;
    font-weight: 720;
    color: #24324a;
}

.section-title .hint {
    margin-left: 8px;
    font-size: 11px;
    font-weight: 400;
    color: var(--text-mute);
}


/* ============================================================
   8. 按钮
   ============================================================ */

.primary-btn {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 650 !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.26) !important;
}

.primary-btn:hover {
    filter: brightness(1.07);
    box-shadow: var(--shadow-lg) !important;
}

.ghost-btn {
    background: #ffffff !important;
    border: 1px solid var(--line-strong) !important;
    color: #46536a !important;
    font-weight: 600 !important;
}

.ghost-btn:hover {
    background: #f1f4fa !important;
    color: #1e2a3a !important;
}

.danger-btn {
    background: #fff1f2 !important;
    border: 1px solid #fecdd3 !important;
    color: #be123c !important;
    font-weight: 650 !important;
}

.danger-btn:hover {
    background: #ffe4e6 !important;
    color: #9f1239 !important;
}


/* ============================================================
   9. 文件上传
   ============================================================ */

.upload-box {
    border: 1px dashed #c7d2fe !important;
    border-radius: var(--radius) !important;

    background: #f8f9ff !important;

    padding: 16px !important;
}

.upload-box .wrap,
.upload-box .file-preview {
    background: transparent !important;
}


/* ============================================================
   10. 文档表格
   ============================================================ */

.table-wrap {
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;

    background: #ffffff;
}

.doc-table {
    width: 100%;
    border-collapse: collapse;

    font-size: 12.5px;
}

.doc-table thead th {
    padding: 12px 14px;

    text-align: left;

    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;

    color: var(--text-mute);

    background: #f8fafc;

    border-bottom: 1px solid var(--line);
}

.doc-table tbody td {
    padding: 12px 14px;

    color: #3b4a61;

    border-bottom: 1px solid #eef1f7;
}

.doc-table tbody tr:last-child td {
    border-bottom: none;
}

.doc-table tbody tr:hover td {
    background: #f8f9ff;
}

.doc-name {
    color: #16203a !important;
    font-weight: 600;
}

.doc-mute {
    color: var(--text-mute) !important;
    font-family: "SFMono-Regular", Consolas, monospace;
}

.tag {
    display: inline-block;

    padding: 3px 9px;

    border-radius: 999px;

    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.4px;
}

.tag-jd {
    background: #eef2ff;
    color: #4338ca;
    border: 1px solid #c7d2fe;
}

.tag-resume {
    background: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}

.empty-box {
    padding: 34px 20px;

    text-align: center;

    border: 1px dashed var(--line-strong);
    border-radius: 12px;

    background: var(--panel-soft);

    color: var(--text-mute);
    font-size: 12.5px;
    line-height: 1.9;
}


/* ============================================================
   11. 流程 / 步骤
   ============================================================ */

.flow {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    align-items: stretch;
}

.flow-step {
    flex: 1 1 150px;

    padding: 14px 16px;

    border: 1px solid var(--line);
    border-radius: 12px;

    background: #ffffff;

    box-shadow: var(--shadow-sm);
}

.flow-index {
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1px;
    color: #6366f1;
}

.flow-name {
    margin-top: 7px;
    font-size: 13.5px;
    font-weight: 700;
    color: #16203a;
}

.flow-desc {
    margin-top: 5px;
    font-size: 11px;
    color: var(--text-mute);
    line-height: 1.6;
}


/* ============================================================
   12. AI Agent
   ============================================================ */

.chat-panel {
    padding: 20px 22px !important;
}

.chat-header {
    margin-bottom: 14px;
}

.chat-header-title {
    font-size: 15px;
    font-weight: 730;
    color: #16203a;
}

.chat-header-desc {
    margin-top: 5px;
    font-size: 11.5px;
    color: var(--text-mute);
}


.chatbot {
    /* 关键：把 Gradio 的配色变量就地覆盖，
       这样聊天区内部的容器、消息区都会跟着变色，
       不会再出现「对话框和白色卡片一个颜色」的问题。 */
    --block-background-fill: #e9effb !important;
    --block-background-fill-dark: #e9effb !important;
    --background-fill-primary: #e9effb !important;
    --background-fill-primary-dark: #e9effb !important;
    --background-fill-secondary: #e9effb !important;
    --background-fill-secondary-dark: #e9effb !important;
    --border-color-primary: #c3d0f0 !important;
    --border-color-primary-dark: #c3d0f0 !important;

    background: #e9effb !important;

    border: 1px solid #c3d0f0 !important;
    border-radius: 14px !important;

    box-shadow: inset 0 2px 6px rgba(15, 23, 42, 0.06);
}

/* 双保险：直接给内部容器也上色 */
.chatbot > div,
.chatbot .panel,
.chatbot .message-wrap,
.chatbot .bubble-wrap {
    background: #e9effb !important;
}

/* 聊天气泡在亮色下的可读性 */
.chatbot .message,
.chatbot .message-row {
    font-size: 14px !important;
}

.chatbot .message.user,
.chatbot .user {
    background: #4f46e5 !important;
    color: #ffffff !important;
}

.chatbot .message.bot,
.chatbot .bot {
    background: #ffffff !important;
    color: #1e2a3a !important;
    border: 1px solid #dbe4f5 !important;
}

/* Agent 主 Row：
   不强制左右两栏等高。 */
.agent-row {
    align-items: flex-start !important;
}

/* Try Asking 一栏：
   白框按内容收缩，不拉伸。

   原因：
     一旦把白框拉伸到和对话框一样高，
     多出来的空间就会变成「Try Asking 头顶上的白框」。
     这里不拉伸，白框从 Try Asking 这一行开始，
     上面不会再顶着一块空白。 */
.agent-side {
    display: flex !important;
    flex-direction: column !important;

    justify-content: flex-start !important;
    align-self: flex-start !important;

    padding: 18px !important;

    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;

    background: var(--panel) !important;

    box-shadow: var(--shadow-sm);
}

.agent-side-title {
    display: flex;
    align-items: center;
    gap: 8px;

    font-size: 13px;
    font-weight: 800;
    letter-spacing: 0.6px;
    text-transform: uppercase;

    color: #4338ca;
}

.agent-side-title::before {
    content: "";

    width: 3px;
    height: 15px;

    border-radius: 2px;

    background: linear-gradient(180deg, #4f46e5, #7c3aed);
}

.agent-side-desc {
    margin-top: 7px;
    margin-bottom: 12px;
    font-size: 11.5px;
    color: #64748b;
    line-height: 1.7;
}

.example-btn {
    width: 100% !important;

    justify-content: flex-start !important;
    text-align: left !important;

    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;

    color: #3730a3 !important;

    font-size: 12px !important;
    font-weight: 650 !important;

    padding: 10px 12px !important;
    margin: 5px 0 !important;

    border-radius: 10px !important;

    box-shadow: 0 1px 2px rgba(79, 70, 229, 0.06);
}

.example-btn:hover {
    background: #eef2ff !important;
    border-color: #c7d2fe !important;
}


/* ============================================================
   13. 设置
   ============================================================ */

.settings-card {
    padding: 22px !important;

    border: 1px solid var(--line) !important;
    border-radius: var(--radius) !important;

    background: var(--panel) !important;

    box-shadow: var(--shadow-sm);
}

.settings-note {
    padding: 12px 14px;

    border-radius: 10px;

    background: #eef2ff;
    border: 1px solid #c7d2fe;

    font-size: 11.5px;
    color: #3730a3;
    line-height: 1.7;
}


/* ============================================================
   14. 表单控件
   ============================================================ */

textarea,
input[type="text"],
input[type="password"] {
    background: #ffffff !important;
    border: 1px solid var(--line-strong) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

textarea::placeholder,
input::placeholder {
    color: #9aa6b8 !important;
}

textarea:focus,
input[type="text"]:focus,
input[type="password"]:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.14) !important;
}

label,
.block-title {
    color: var(--text-dim) !important;
    font-size: 11.5px !important;
    font-weight: 600 !important;
}


/* ============================================================
   15. Markdown
   ============================================================ */

.prose {
    color: #2b3a52 !important;
}

.prose h2,
.prose h3 {
    color: #16203a !important;
}

.prose a {
    color: #4f46e5 !important;
}

.prose code {
    background: #eef2ff !important;
    color: #4338ca !important;
    border-radius: 5px;
    padding: 1px 5px;
}

.prose pre {
    background: #f8fafc !important;
    border: 1px solid var(--line) !important;
}

.prose table {
    background: #ffffff !important;
}

.prose th {
    background: #f8fafc !important;
    color: #24324a !important;
}


/* ============================================================
   15.45 常驻小提示
   ============================================================ */

.inline-tip {
    margin-bottom: 14px;
    padding: 10px 13px;

    border-radius: 9px;

    background: #fffbeb;
    border: 1px solid #fde68a;

    font-size: 11.5px;
    line-height: 1.7;
    color: #92400e;
}


/* ============================================================
   15.35 简历多选：无论选多少份，高度都固定
   ============================================================ */

/* 撑高的真正原因：
   Gradio 的 .wrap-inner 上有 flex-wrap: wrap，
   每个已选项都是一个标签，选几百份时标签不断换行，
   容器高度就跟着一路涨。

   这里限制标签区高度，超出部分在框内滚动，
   整栏高度就恒定不变了。 */
/* 三重保险：
   不管 Gradio 内部具体是哪个容器在长高，
   这几个都锁死，确保整栏高度恒定。 */
.resume-select .wrap,
.resume-select .wrap-inner,
.resume-select .secondary-wrap,
.doc-select .wrap,
.doc-select .wrap-inner,
.doc-select .secondary-wrap {
    max-height: 84px !important;

    /* 保留换行（标签才够得着），
       但用固定的最大高度 + 垂直滚动，
       选多少份都不会把整栏撑高。 */
    overflow-y: auto !important;
    overflow-x: hidden !important;

    flex-wrap: wrap !important;
}

/* 最外层那个带边框的输入框，高度也锁住 */
.resume-select .wrap,
.doc-select .wrap {
    max-height: 88px !important;
}


.confirm-cb,
.confirm-cb label,
.confirm-cb label span,
.confirm-cb > div,
.confirm-cb > span {
    background: #ffffff !important;
    color: #4b5a72 !important;

    font-size: 12px !important;
    font-weight: 650 !important;
}

.confirm-cb input[type="checkbox"] {
    --checkbox-background-color-selected: #e11d48 !important;
    --checkbox-border-color-selected: #e11d48 !important;
    --checkbox-border-color: #d7a5ae !important;

    width: 16px !important;
    height: 16px !important;

    cursor: pointer !important;
}


/* ============================================================
   15.36 全选勾选框
   ============================================================ */

/* 和下拉框的输入框底部对齐 */
.select-all-cb {
    align-self: flex-end !important;
    margin-bottom: 4px !important;

    min-width: 78px !important;
    padding: 8px 10px !important;

    border: 1px solid #d7deeb !important;
    border-radius: 9px !important;

    background: #ffffff !important;
}

/* 关键：覆盖 Gradio 默认的灰色底 */
.select-all-cb,
.select-all-cb label,
.select-all-cb label span,
.select-all-cb > div,
.select-all-cb > span {
    background: #ffffff !important;
    color: #4b5a72 !important;

    font-size: 12px !important;
    font-weight: 650 !important;
}

/* 勾选框本体：
   Gradio 用 --checkbox-* 变量控制颜色，
   默认的边框是灰的（用户看到的就是这个灰）。
   这里全部换成品牌色系。 */
.select-all-cb input[type="checkbox"] {
    --checkbox-background-color: #ffffff !important;
    --checkbox-background-color-hover: #f4f6fb !important;
    --checkbox-background-color-focus: #ffffff !important;
    --checkbox-background-color-selected: #4f46e5 !important;

    --checkbox-border-color: #c3cbdb !important;
    --checkbox-border-color-hover: #9aa6b8 !important;
    --checkbox-border-color-focus: #818cf8 !important;
    --checkbox-border-color-selected: #4f46e5 !important;

    --checkbox-shadow: none !important;

    width: 16px !important;
    height: 16px !important;

    cursor: pointer !important;
}

/* 勾上以后文字也变成品牌色，状态更明显 */
.select-all-cb input[type="checkbox"]:checked ~ .label-text {
    color: #4f46e5 !important;
    font-weight: 700 !important;
}


/* ============================================================
   15.4 分析对象栏的快捷按钮 / 范围徽章
   ============================================================ */

/* 全选 / 清空 / 刷新：做成小巧的白色按钮 */
.mini-btn {
    font-size: 11.5px !important;
    font-weight: 650 !important;

    padding: 6px 8px !important;
    min-height: 38px !important;

    border-radius: 9px !important;

    background: #ffffff !important;
    border: 1px solid #d7deeb !important;
    color: #4b5a72 !important;

    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
}

.mini-btn:hover {
    background: #eef2ff !important;
    border-color: #b9c5da !important;
    color: #4338ca !important;
}

/* 当前范围：做成一条浅色徽章，比裸文字更清楚 */
.scope-hint {
    margin-top: 4px !important;

    padding: 7px 12px;

    border-radius: 8px;

    background: rgba(255, 255, 255, 0.78);
    border: 1px solid #dbe1fb;
}

.scope-hint p {
    margin: 0 !important;

    font-size: 11.5px !important;
    line-height: 1.6 !important;
    color: #4b5a72 !important;
}


/* ============================================================
   15.5 分析对象（嵌在对话框内部）
   ============================================================ */

/* 做成一张浅色小卡片：
   渐变底 + 圆角 + 内边距，
   比原来那条细线更醒目、也更宽裕 */
.picker-inline {
    /* 只做「宽」，不做「高」：
       内边距保持紧凑，不额外增加高度 */
    padding: 12px 16px !important;
    margin-bottom: 16px;

    border: 1px solid #dbe1fb;
    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #f6f8ff 0%,
            #eef2ff 100%
        ) !important;
}

/* 标题区 */
.picker-head {
    display: flex;
    align-items: center;
    gap: 9px;

    margin-bottom: 10px;
}

.picker-head-bar {
    width: 4px;
    height: 17px;
    flex: 0 0 4px;

    border-radius: 2px;

    background: linear-gradient(180deg, #4f46e5, #7c3aed);
}

.picker-head-title {
    font-size: 13px;
    font-weight: 760;
    color: #24324a;
}

/* 说明文字和标题并排，不占额外一行高度 */
.picker-head-desc {
    font-size: 11px;
    color: #8b98ac;
}


/* ============================================================
   15.6 录入方式
   ============================================================ */

.input-mode {
    font-size: 10px;
    font-weight: 750;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 8px;
}

.divider {
    display: flex;
    align-items: center;
    gap: 12px;

    margin: 18px 0 14px 0;

    font-size: 11px;
    color: var(--text-mute);
}

.divider::before,
.divider::after {
    content: "";
    flex: 1;
    height: 1px;
    background: var(--line-strong);
}


/* ============================================================
   16. 响应式
   ============================================================ */

@media (max-width: 980px) {

    .app-row {
        flex-wrap: wrap !important;
    }

    .sidebar {
        position: static !important;

        flex: 1 1 100% !important;
        width: 100% !important;
        max-width: 100% !important;
        min-width: 0 !important;

        height: auto;
        max-height: none;

        border-right: none !important;
        border-bottom: 1px solid var(--line) !important;
    }

    .app-row {
        padding-left: 0 !important;
    }

    .main {
        margin-left: 0 !important;
        padding: 22px 18px 40px 18px !important;
    }

    .page-title {
        font-size: 22px;
    }
}"""

# ============================================================
# 四、通用小组件
# ============================================================

def _escape(text) -> str:
    """
    转义 HTML，避免文件名中的特殊字符破坏页面。
    """

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _page_head(
    eyebrow: str,
    title: str,
    desc: str,
) -> str:
    """
    统一的页面标题。
    """

    return f"""
    <div class="page-head">
        <div class="page-eyebrow">{_escape(eyebrow)}</div>
        <div class="page-title">{_escape(title)}</div>
        <div class="page-desc">{_escape(desc)}</div>
    </div>
    """


def _stat_card(
    label: str,
    value,
    sub: str,
    tone: str = "indigo",
) -> str:
    """
    数据卡片。

    tone 决定左侧色条：
        indigo / violet / emerald / amber
    """

    return f"""
    <div class="stat-card tone-{tone}">
        <div class="stat-label">{_escape(label)}</div>
        <div class="stat-value">{_escape(value)}</div>
        <div class="stat-sub">{_escape(sub)}</div>
    </div>
    """


def _kind_label(doc_type: str) -> str:
    """
    doc_type → 中文标签。
    """

    return "JD" if doc_type == "jd" else "简历"


def _kind_tag(doc_type: str) -> str:
    """
    doc_type → 彩色标签 HTML。
    """

    if doc_type == "jd":

        return '<span class="tag tag-jd">JD</span>'

    return '<span class="tag tag-resume">简历</span>'


def _doc_table(
    documents: list[dict],
    empty_text: str = "还没有任何文档",
) -> str:
    """
    把文档列表渲染成表格 HTML。
    """

    if not documents:

        return f"""
        <div class="empty-box">
            {_escape(empty_text)}<br>
            上传 PDF / DOCX 后，这里会显示文档列表。
        </div>
        """

    rows = []

    for document in documents:

        doc_id = str(document.get("doc_id", ""))

        short_id = (
            doc_id[:8] + "…"
            if len(doc_id) > 8
            else doc_id
        )

        rows.append(
            f"""
            <tr>
                <td>{_kind_tag(document.get("doc_type"))}</td>
                <td class="doc-name">
                    {_escape(document.get("filename", "未命名"))}
                </td>
                <td>{document.get("chunk_count", 0)}</td>
                <td class="doc-mute">{_escape(short_id)}</td>
            </tr>
            """
        )

    return f"""
    <div class="table-wrap">
        <table class="doc-table">
            <thead>
                <tr>
                    <th>类型</th>
                    <th>文件名</th>
                    <th>Chunk</th>
                    <th>文档 ID</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def _doc_choices(
    documents: list[dict],
) -> list[tuple[str, str]]:
    """
    把文档列表转换成下拉框选项。

    Gradio Dropdown 支持：

        [(显示文本, 实际值), ...]
    """

    return [
        (
            f"[{_kind_label(document.get('doc_type'))}] "
            f"{document.get('filename', '未命名')}"
            f" · {document.get('chunk_count', 0)} chunks",

            document.get("doc_id", ""),
        )
        for document in documents
        if document.get("doc_id")
    ]


# ============================================================
# 五、刷新函数（供页面和 app.load 使用）
# ============================================================

def refresh_dashboard() -> tuple:
    """
    重新读取 Dashboard 统计，返回 4 张卡片。
    """

    try:

        stats = get_dashboard_stats()

    except Exception as exc:

        print("Dashboard refresh error:", exc)

        stats = {
            "jd_count": 0,
            "resume_count": 0,
            "chunk_count": 0,
            "document_count": 0,
        }

    return (
        _stat_card(
            "JD 岗位",
            stats.get("jd_count", 0),
            "已进入 JD 知识库",
            "indigo",
        ),

        _stat_card(
            "候选人简历",
            stats.get("resume_count", 0),
            "已进入简历知识库",
            "violet",
        ),

        _stat_card(
            "向量片段",
            stats.get("chunk_count", 0),
            "Chroma Chunks",
            "emerald",
        ),

        _stat_card(
            "文档总数",
            stats.get("document_count", 0),
            "JD + 简历",
            "amber",
        ),
    )


def refresh_documents(
    doc_type: str = "both",
) -> tuple:
    """
    重新读取文档列表。

    返回：

        (表格 HTML, 下拉框更新)
    """

    try:

        documents = get_documents(doc_type)

    except Exception as exc:

        print("Document refresh error:", exc)

        documents = []

    return (
        _doc_table(
            documents,
            "还没有文档",
        ),
        gr.update(
            choices=_doc_choices(documents),
            value=[],
        ),
    )


# ============================================================
# 六、概览页
# ============================================================

def overview_page() -> tuple:
    """
    概览页。

    返回需要刷新 / 绑定的组件：

        (jd_card, resume_card, chunk_card, document_card,
         quick_jd_btn, quick_resume_btn, quick_agent_btn)
    """

    gr.HTML(
        _page_head(
            "Overview",
            "招聘分析概览",
            "统一查看知识库规模、RAG 结构与常用入口。",
        )
    )

    # --------------------------------------------------------
    # 统计卡片
    # --------------------------------------------------------

    with gr.Row():

        jd_card = gr.HTML(
            _stat_card(
                "JD 岗位",
                "–",
                "已进入 JD 知识库",
                "indigo",
            )
        )

        resume_card = gr.HTML(
            _stat_card(
                "候选人简历",
                "–",
                "已进入简历知识库",
                "violet",
            )
        )

        chunk_card = gr.HTML(
            _stat_card(
                "向量片段",
                "–",
                "Chroma Chunks",
                "emerald",
            )
        )

        document_card = gr.HTML(
            _stat_card(
                "文档总数",
                "–",
                "JD + 简历",
                "amber",
            )
        )

    # --------------------------------------------------------
    # RAG 流程
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="section-title">
            RAG Pipeline
            <span class="hint">从文档到 Agent 的完整链路</span>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="flow">

            <div class="flow-step">
                <div class="flow-index">01</div>
                <div class="flow-name">Document</div>
                <div class="flow-desc">PDF / DOCX 的 JD 与简历</div>
            </div>

            <div class="flow-step">
                <div class="flow-index">02</div>
                <div class="flow-name">Chunk</div>
                <div class="flow-desc">递归切分为语义文本块</div>
            </div>

            <div class="flow-step">
                <div class="flow-index">03</div>
                <div class="flow-name">BGE-M3</div>
                <div class="flow-desc">本地生成稠密向量</div>
            </div>

            <div class="flow-step">
                <div class="flow-index">04</div>
                <div class="flow-name">Chroma</div>
                <div class="flow-desc">持久化向量数据库</div>
            </div>

            <div class="flow-step">
                <div class="flow-index">05</div>
                <div class="flow-name">Retrieval</div>
                <div class="flow-desc">语义检索相关片段</div>
            </div>

            <div class="flow-step">
                <div class="flow-index">06</div>
                <div class="flow-name">Agent</div>
                <div class="flow-desc">LangGraph 工具调用与匹配</div>
            </div>

        </div>
        """
    )

    # --------------------------------------------------------
    # 快速开始
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="section-title">
            快速开始
        </div>
        """
    )

    with gr.Row():

        quick_jd_btn = gr.Button(
            "▤  导入岗位 JD",
            elem_classes=["ghost-btn"],
        )

        quick_resume_btn = gr.Button(
            "❏  导入候选人简历",
            elem_classes=["ghost-btn"],
        )

        quick_agent_btn = gr.Button(
            "✦  开始 AI 分析",
            elem_classes=["primary-btn"],
        )

    return (
        jd_card,
        resume_card,
        chunk_card,
        document_card,
        quick_jd_btn,
        quick_resume_btn,
        quick_agent_btn,
    )


# ============================================================
# 七、文档管理页（JD / 简历共用）
# ============================================================

def document_page(
    prefix: str,
    doc_type: str,
    eyebrow: str,
    title: str,
    desc: str,
    hint: str,
) -> tuple:
    """
    JD 管理与简历库共用的页面。

    参数：

        prefix
            用于生成唯一 elem_id，避免两个页面冲突

        doc_type
            "jd" 或 "resume"

    返回：

        (表格, 下拉框)

    说明：
        导入 / 删除的结果通过 Toast 弹窗提示（几秒后自动消失），
        不再在页面上留一大块常驻文字。
    """

    kind = _kind_label(doc_type)

    empty_text = (
        "还没有 JD 文档"
        if doc_type == "jd"
        else "还没有候选人简历"
    )

    gr.HTML(
        _page_head(
            eyebrow,
            title,
            desc,
        )
    )

    with gr.Row():

        # ----------------------------------------------------
        # 左：录入
        # ----------------------------------------------------

        with gr.Column(
            scale=2,
            elem_classes=["panel"],
        ):

            gr.HTML(
                f"""
                <div class="section-title" style="margin-top:0;">
                    录入{_escape(kind)}
                </div>
                <div class="page-desc" style="margin-bottom:12px;">
                    {_escape(hint)}
                </div>
                """
            )

            # 常驻小提示：说明什么情况会失败、怎么绕过。
            gr.HTML(
                """
                <div class="inline-tip">
                    扫描件 / 图片型 PDF 没有文字层，提取不到内容，
                    请改用下面的「方式二」直接粘贴文本。
                </div>
                """
            )

            # ------------------------------------------------
            # 方式一：上传 PDF / DOCX
            # ------------------------------------------------

            with gr.Column(
                elem_classes=["upload-box"],
            ):

                gr.HTML(
                    '<div class="input-mode">方式一 · 上传文件</div>'
                )

                files = gr.File(
                    label=f"选择{kind}文件（可多选）",
                    file_count="multiple",
                    file_types=[".pdf", ".docx"],
                    elem_id=f"{prefix}-files",
                )

                upload_btn = gr.Button(
                    f"导入{kind}知识库",
                    elem_classes=["primary-btn"],
                )

            # ------------------------------------------------
            # 方式二：直接粘贴 / 输入文本
            # ------------------------------------------------

            gr.HTML(
                """
                <div class="divider">
                    <span>或直接输入文本</span>
                </div>
                """
            )

            gr.HTML(
                '<div class="input-mode">方式二 · 直接输入</div>'
            )

            title_input = gr.Textbox(
                label="标题（可选）",
                placeholder=(
                    "例如：AI产品经理 JD"
                    if doc_type == "jd"
                    else "例如：张三 - 简历"
                ),
                lines=1,
                elem_id=f"{prefix}-title",
            )

            text_input = gr.Textbox(
                label=f"{kind}正文",
                placeholder=(
                    "直接粘贴或输入内容…\n\n"
                    "例如：\n"
                    "岗位名称：AI产品经理\n"
                    "岗位职责：\n"
                    "1. 负责……\n"
                    "任职要求：\n"
                    "1. 熟悉……"
                ),
                lines=12,
                elem_id=f"{prefix}-text",
            )

            text_btn = gr.Button(
                f"导入文本{kind}",
                elem_classes=["primary-btn"],
            )

        # ----------------------------------------------------
        # 右：列表 + 删除
        # ----------------------------------------------------

        with gr.Column(
            scale=3,
            elem_classes=["panel"],
        ):

            gr.HTML(
                f"""
                <div class="section-title" style="margin-top:0;">
                    已入库的{_escape(kind)}
                    <span class="hint">
                        删除会同时移除向量与原文件
                    </span>
                </div>
                """
            )

            table = gr.HTML(
                _doc_table([], empty_text)
            )

            gr.HTML(
                '<div class="input-mode">批量删除</div>'
            )

            dropdown = gr.Dropdown(
                label="选择要删除的文档（可多选）",
                choices=[],
                value=[],
                multiselect=True,
                interactive=True,
                elem_id=f"{prefix}-dropdown",
                elem_classes=["doc-select"],
            )

            with gr.Row():

                delete_btn = gr.Button(
                    "🗑  删除选中",
                    elem_classes=["danger-btn"],
                    scale=1,
                )

                refresh_btn = gr.Button(
                    "↻  刷新",
                    elem_classes=["ghost-btn"],
                    scale=1,
                )

            with gr.Row():

                confirm_all_cb = gr.Checkbox(
                    label="确认清空",
                    value=False,
                    scale=1,
                    elem_classes=["confirm-cb"],
                )

                delete_all_btn = gr.Button(
                    "⚠  全部删除",
                    elem_classes=["danger-btn"],
                    scale=1,
                )

    # --------------------------------------------------------
    # 事件处理
    # --------------------------------------------------------

    def current_outputs():
        """
        刷新文档表格与删除下拉框。
        """

        docs = get_documents(doc_type)

        return (
            _doc_table(docs, empty_text),
            gr.update(
                choices=_doc_choices(docs),
                value=[],
            ),
        )

    def do_upload(selected_files):

        if not selected_files:

            gr.Warning("请先选择文件。", duration=4)

            return current_outputs()

        try:

            result = ingest_files(
                selected_files,
                doc_type,
            )

            if result["failed"]:

                gr.Warning(
                    result["summary"],
                    duration=9,
                )

            else:

                gr.Info(
                    result["summary"],
                    duration=4,
                )

        except Exception as exc:

            gr.Warning(
                f"导入失败：{exc}",
                duration=9,
            )

        return current_outputs()

    def do_text_import(title, content):

        try:

            result = ingest_text(
                content,
                doc_type,
                title,
            )

            name = result.splitlines()[0]
            name = name.split("：", 1)[-1]

            gr.Info(
                f"已导入：{name}",
                duration=4,
            )

            # 成功后清空输入框，避免重复导入。
            remain = ""

        except Exception as exc:

            gr.Warning(
                f"导入失败：{exc}",
                duration=9,
            )

            # 失败时保留内容，避免用户输入丢失。
            remain = content

        return current_outputs() + (remain,)

    def do_delete(doc_ids):
        """
        批量删除选中的文档。
        """

        if not doc_ids:

            gr.Warning("请先选择要删除的文档。", duration=4)

            return current_outputs()

        ok = 0
        failed = 0

        for doc_id in doc_ids:

            message = delete_document(doc_id)

            if message.startswith("✅"):

                ok += 1

            else:

                failed += 1

        if failed:

            gr.Warning(
                f"删除完成：成功 {ok} 个，失败 {failed} 个",
                duration=7,
            )

        else:

            gr.Info(
                f"已删除 {ok} 个文档",
                duration=4,
            )

        return current_outputs()

    def do_delete_all(confirmed):
        """
        清空当前知识库里的全部文档。

        必须先勾选「确认清空」，避免误操作。
        """

        if not confirmed:

            gr.Warning(
                "请先勾选「确认清空」再点全部删除。",
                duration=5,
            )

            return current_outputs() + (
                gr.update(value=False),
            )

        docs = get_documents(doc_type)

        if not docs:

            gr.Warning(
                f"{kind}库已经是空的。",
                duration=4,
            )

            return current_outputs() + (
                gr.update(value=False),
            )

        ok = 0

        for item in docs:

            message = delete_document(item["doc_id"])

            if message.startswith("✅"):

                ok += 1

        gr.Info(
            f"已清空全部 {ok} 个文档",
            duration=4,
        )

        return current_outputs() + (
            gr.update(value=False),
        )

    # --------------------------------------------------------
    # 事件绑定
    # --------------------------------------------------------

    upload_btn.click(
        do_upload,
        inputs=[files],
        outputs=[
            table,
            dropdown,
        ],
    )

    text_btn.click(
        do_text_import,
        inputs=[
            title_input,
            text_input,
        ],
        outputs=[
            table,
            dropdown,
            text_input,
        ],
    )

    delete_btn.click(
        do_delete,
        inputs=[dropdown],
        outputs=[
            table,
            dropdown,
        ],
    )

    delete_all_btn.click(
        do_delete_all,
        inputs=[confirm_all_cb],
        outputs=[
            table,
            dropdown,
            confirm_all_cb,
        ],
    )

    refresh_btn.click(
        lambda: refresh_documents(doc_type),
        outputs=[
            table,
            dropdown,
        ],
    )

    return (
        table,
        dropdown,
    )

# ============================================================
# 八、AI Agent 对话页
# ============================================================

# Try Asking 示例
#
# 每个元素是 (按钮上的短文字, 实际填入输入框的完整问题)
#
# 用短文字是为了让按钮能排成一行，
# 放在「发送」按钮旁边，符合「先选示例、再发送」的操作习惯。
# Try Asking 示例问题
#
# 在右侧竖排成「一栏」，点击即填入输入框。
AGENT_EXAMPLES = [
    "帮我筛选最适合这个岗位的候选人，并给出排序",
    "这份简历适合哪些岗位？",
    "对比这些候选人，说明优势与不足",
    "分析这个岗位最看重哪些技能",
]


def _load_doc_map(doc_type: str) -> dict:
    """
    读取某类文档，返回 {doc_id: 文档信息} 的字典。
    """

    try:

        return {
            item["doc_id"]: item
            for item in get_documents(doc_type)
        }

    except Exception as exc:

        print("Load doc map error:", exc)

        return {}


def _normalize_jd(jd_value):
    """
    统一 JD 选择的值。

    JD 下拉现在是「多选 + 最多选 1 个」，
    这样才会有和简历一样的小叉可以取消；
    但它的值是列表，这里统一取出单个 doc_id。
    """

    if isinstance(jd_value, (list, tuple)):

        return jd_value[0] if jd_value else None

    return jd_value


def build_agent_context(
    jd_id,
    resume_ids,
) -> str:
    """
    把用户选中的岗位 / 简历转换成给 Agent 的「限定范围」说明。

    这样提问就会绑定到具体的文档上，
    Agent 不会在整个知识库里盲搜。
    """

    jd_id = _normalize_jd(jd_id)

    jds = _load_doc_map("jd")
    resumes = _load_doc_map("resume")

    lines = []

    if jd_id and jd_id in jds:

        lines.append(
            f"- 岗位 JD：《{jds[jd_id]['filename']}》"
            f"  document_id = {jd_id}"
        )

    for resume_id in (resume_ids or []):

        if resume_id in resumes:

            lines.append(
                f"- 候选人简历：《{resumes[resume_id]['filename']}》"
                f"  document_id = {resume_id}"
            )

    if not lines:

        return ""

    return (
        "【本次任务的限定范围】\n"
        "用户已在界面中选定以下知识库文档，"
        "请严格围绕这些文档作答：\n\n"
        + "\n".join(lines)
        + "\n\n"
        "要求：\n"
        "1. 先调用 get_document 工具读取上述 document_id 的完整内容，"
        "再进行分析与回答。\n"
        "2. 不要使用未列出的其他 JD / 简历，除非用户明确要求。\n"
        "3. 如果用户的问题与所选文档无关，请提醒用户。"
    )


def agent_scope_text(
    jd_id,
    resume_ids,
) -> str:
    """
    把当前选择渲染成一行提示文字，显示在 Agent 页顶部。
    """

    jd_id = _normalize_jd(jd_id)

    jds = _load_doc_map("jd")
    resumes = _load_doc_map("resume")

    parts = []

    if jd_id and jd_id in jds:

        parts.append(
            f"岗位《{jds[jd_id]['filename']}》"
        )

    picked = [
        resumes[item]["filename"]
        for item in (resume_ids or [])
        if item in resumes
    ]

    if picked:

        # 选得少就列名字，选得多就用「N 份」概括，
        # 避免提示文字太长。
        if len(picked) <= 2:

            names = "、".join(
                f"《{name}》"
                for name in picked
            )

        else:

            names = f"《{picked[0]}》等 {len(picked)} 份"

        parts.append(f"简历 {names}")

    if not parts:

        return (
            "**当前范围：** 未选择岗位 / 简历，"
            "Agent 将在整个知识库中检索。"
        )

    return "**当前范围：** " + " ｜ ".join(parts)


def refresh_agent_selectors(
    current_jd=None,
    current_resumes=None,
) -> tuple:
    """
    刷新 Agent 页的岗位 / 简历下拉框。

    返回：

        (岗位下拉 update, 简历下拉 update, 范围提示,
         全选勾选框 update)

    已经选中且仍然存在的文档会被保留，
    被删除的文档会自动取消选择。
    """

    jd_documents = _load_doc_map("jd")
    resume_documents = _load_doc_map("resume")

    jd_choices = _doc_choices(list(jd_documents.values()))
    resume_choices = _doc_choices(list(resume_documents.values()))

    # JD 是多选组件，值必须是列表。
    current_jd = _normalize_jd(current_jd)

    jd_value = (
        [current_jd]
        if current_jd in jd_documents
        else []
    )

    resume_values = [
        item
        for item in (current_resumes or [])
        if item in resume_documents
    ]

    # 「全选」勾选框的状态：简历全部选满才算勾上。
    all_resume_ids = set(resume_documents.keys())

    is_all_selected = (
        bool(all_resume_ids)
        and set(resume_values) == all_resume_ids
    )

    return (
        gr.update(
            choices=jd_choices,
            value=jd_value,
        ),
        gr.update(
            choices=resume_choices,
            value=resume_values,
        ),
        agent_scope_text(jd_value, resume_values),
        gr.update(value=is_all_selected),
    )


def agent_page() -> tuple:
    """
    AI Agent 页面。

    布局：

        ┌───────────────────────────────┬──────────────┐
        │ 对话 (panel)                   │ Try Asking    │
        │  ┌─ 分析对象 ───────────────┐  │  示例 1       │
        │  │ [岗位JD▼][简历▼][清空][刷新]│  │  示例 2       │
        │  │ 当前范围：……               │  │  示例 3       │
        │  └──────────────────────────┘  │  示例 4       │
        │  ┌──────────────────────────┐  │  清空对话     │
        │  │        聊天记录            │  │              │
        │  └──────────────────────────┘  │              │
        │  [输入框................] [发送] │              │
        └───────────────────────────────┴──────────────┘

    要点：

        1. 分析对象嵌在对话框内部的最上方，不额外占一行高度。
        2. Try Asking 是右侧竖排一栏，紧挨对话框。
        3. 对话框始终是主区域。

    返回：

        (chatbot, textbox, status,
         jd_select, resume_select, scope_hint, refresh_btn)
    """

    gr.HTML(
        _page_head(
            "Agent",
            "AI 招聘分析 Agent",
            "选定分析对象后直接提问，Agent 只围绕这些文档工作。",
        )
    )

    # ========================================================
    # 1. 分析对象：通栏
    #
    #    横跨整个内容宽度。
    #    这样一来，右边那一截正好补上
    #    Try Asking 上方原本空着的位置。
    # ========================================================

    with gr.Column(
        elem_classes=["panel", "picker-inline", "picker-full"],
    ):

        gr.HTML(
            """
            <div class="picker-head">
                <div class="picker-head-bar"></div>
                <div class="picker-head-title">
                    分析对象
                </div>
                <div class="picker-head-desc">
                    选定后，Agent 只围绕这些文档工作
                </div>
            </div>
            """
        )

        with gr.Row():

            # 用「多选 + 最多 1 个」的方式，
            # 这样选完之后旁边会出现和简历一样的小叉，
            # 可以一键取消选择。
            jd_select = gr.Dropdown(
                label="岗位 JD",
                choices=[],
                value=[],
                multiselect=True,
                max_choices=1,
                interactive=True,
                scale=3,
                elem_id="agent-jd-select",
            )

            resume_select = gr.Dropdown(
                label="候选人简历（可多选）",
                choices=[],
                value=[],
                multiselect=True,
                interactive=True,
                scale=4,
                elem_id="agent-resume-select",
                elem_classes=["resume-select"],
            )

            # 用勾选框代替独立按钮：
            # 勾上 = 选中全部简历；取消 = 清空简历选择。
            select_all_cb = gr.Checkbox(
                label="全选",
                value=False,
                scale=1,
                elem_classes=["select-all-cb"],
            )

            clear_select_btn = gr.Button(
                "✕ 清空",
                elem_classes=["mini-btn"],
                scale=1,
            )

            refresh_btn = gr.Button(
                "↻ 刷新",
                elem_classes=["mini-btn"],
                scale=1,
            )

        scope_hint = gr.Markdown(
            "**当前范围：** 未选择岗位 / 简历，"
            "Agent 将在整个知识库中检索。",
            elem_classes=["scope-hint"],
        )

    # ========================================================
    # 2. 对话区（左） + Try Asking（右）
    # ========================================================

    with gr.Row(
        elem_classes=["agent-row"],
    ):

        # ------------------------------------------------
        # 左：对话区
        # ------------------------------------------------

        with gr.Column(
            scale=4,
            elem_classes=["panel", "chat-panel"],
        ):

            chatbot = gr.Chatbot(
                height=470,
                show_label=False,
                elem_classes=["chatbot"],
            )

            with gr.Row():

                textbox = gr.Textbox(
                    placeholder=(
                        "输入招聘任务，按 Enter 发送…"
                    ),
                    lines=2,
                    show_label=False,
                    container=False,
                    scale=5,
                )

                send_btn = gr.Button(
                    "发送",
                    elem_classes=["primary-btn"],
                    scale=1,
                )

        # ------------------------------------------------
        # 右：Try Asking（竖排一栏）
        # ------------------------------------------------

        with gr.Column(
            scale=1,
            elem_classes=["agent-side"],
        ):

            gr.HTML(
                """
                <div class="agent-side-title">
                    Try Asking
                </div>
                <div class="agent-side-desc">
                    点击示例会直接填入输入框
                </div>
                """
            )

            for example in AGENT_EXAMPLES:

                example_btn = gr.Button(
                    example,
                    elem_classes=["example-btn"],
                )

                example_btn.click(
                    lambda value=example: value,
                    outputs=[textbox],
                )

            clear_btn = gr.Button(
                "清空对话",
                elem_classes=["ghost-btn"],
            )

    # --------------------------------------------------------
    # 回复逻辑
    # --------------------------------------------------------

    # --------------------------------------------------------

    def respond(history, message, jd_id, resume_ids):
        """
        生成器版本 —— 分两步更新界面：

            第 1 步：立刻把用户的消息显示到对话框
            第 2 步：Agent 回复后，再追加 AI 的消息

        这样做的好处：

            - 点完发送马上就能看到自己的消息，
              不会出现「点了半天没反应」的错觉
            - 配合外层的「锁定输入」，
              彻底避免并发提交把前面的对话覆盖掉
        """

        history = list(history or [])

        message = (message or "").strip()

        scope = agent_scope_text(jd_id, resume_ids)

        if not message:

            gr.Warning("请输入招聘任务。", duration=3)

            yield history, "", scope

            return

        # ------------------------------------------------
        # 第 1 步：先把用户消息显示出来
        # ------------------------------------------------

        history.append(
            {
                "role": "user",
                "content": message,
            }
        )

        yield history, "", scope

        # ------------------------------------------------
        # 第 2 步：调用 Agent 生成回复
        # ------------------------------------------------

        if not get_api_key().strip():

            answer = (
                "### ⚠️ 尚未配置 API Key\n\n"
                "请先进入 **系统设置**，"
                "填写并保存 DeepSeek API Key。"
            )

        else:

            context = build_agent_context(jd_id, resume_ids)

            try:

                # 传 history[:-1]：
                # 刚才那条用户消息由 chat() 自己追加，
                # 这里不重复传，否则模型会看到两遍。
                answer = chat(
                    message,
                    history[:-1],
                    context=context,
                )

            except Exception as exc:

                answer = (
                    "### ❌ Agent 运行失败\n\n"
                    f"`{type(exc).__name__}`\n\n"
                    f"{exc}"
                )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        yield history, "", scope

    # --------------------------------------------------------
    # 事件绑定
    # --------------------------------------------------------
    #
    # 这里用「锁定 → 处理 → 解锁」三段式。
    #
    # 为什么要锁定？
    #
    #   Gradio 在「点击的那一刻」就会捕获输入组件的值。
    #   而 Agent 一次要跑十几秒，
    #   如果这期间再发一条消息，第二条拿到的还是旧历史，
    #   等它返回时就会把前一条的对话覆盖掉 ——
    #   表现就是「聊天内容莫名其妙消失了」。
    #
    #   所以处理期间把输入框和发送键锁住。
    # --------------------------------------------------------

    def lock_inputs():

        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def unlock_inputs():

        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    # 更新完成后把聊天区滚到底部。
    #
    # 为什么需要？
    #   Gradio 会整体重画消息列表，
    #   重画时滚动位置可能被重置到顶部，
    #   看起来就像「跳回到上面的旧消息」。
    #   这里在渲染后补几次滚动，确保停在最新消息。
    SCROLL_TO_BOTTOM_JS = """
    () => {
        const goBottom = () => {
            const box = document.querySelector('.chatbot .bubble-wrap');
            if (box) { box.scrollTop = box.scrollHeight; }
        };
        setTimeout(goBottom, 40);
        setTimeout(goBottom, 160);
        setTimeout(goBottom, 400);
    }
    """

    for trigger in (textbox.submit, send_btn.click):

        trigger(
            lock_inputs,
            outputs=[send_btn, textbox],
        ).then(
            respond,
            inputs=[chatbot, textbox, jd_select, resume_select],
            outputs=[chatbot, textbox, scope_hint],
        ).then(
            unlock_inputs,
            outputs=[send_btn, textbox],
            js=SCROLL_TO_BOTTOM_JS,
        )
    # 切换选择时实时更新范围提示。
    jd_select.change(
        agent_scope_text,
        inputs=[jd_select, resume_select],
        outputs=[scope_hint],
    )

    resume_select.change(
        agent_scope_text,
        inputs=[jd_select, resume_select],
        outputs=[scope_hint],
    )

    def do_refresh_selectors(jd_id, resume_ids):

        return refresh_agent_selectors(jd_id, resume_ids)

    refresh_btn.click(
        do_refresh_selectors,
        inputs=[jd_select, resume_select],
        outputs=[
            jd_select,
            resume_select,
            scope_hint,
            select_all_cb,
        ],
    )

    def do_toggle_select_all(checked, jd_value):
        """
        勾上「全选」→ 选中简历库里的全部简历；
        取消勾选 → 清空简历选择。

        岗位的选择不受影响。
        """

        # ------------------------------------------------
        # 取消勾选：清空简历
        # ------------------------------------------------

        if not checked:

            return (
                gr.update(value=[]),
                agent_scope_text(jd_value, []),
                gr.update(value=False),
            )

        # ------------------------------------------------
        # 勾上：全选
        # ------------------------------------------------

        resumes = get_documents("resume")

        ids = [
            item["doc_id"]
            for item in resumes
            if item.get("doc_id")
        ]

        if not ids:

            gr.Warning(
                "简历库还是空的，请先到「简历库」上传简历。",
                duration=4,
            )

            return (
                gr.update(value=[]),
                agent_scope_text(jd_value, []),
                gr.update(value=False),
            )

        gr.Info(
            f"已选中全部 {len(ids)} 份简历",
            duration=3,
        )

        return (
            gr.update(
                choices=_doc_choices(resumes),
                value=ids,
            ),
            agent_scope_text(jd_value, ids),
            gr.update(value=True),
        )

    select_all_cb.change(
        do_toggle_select_all,
        inputs=[select_all_cb, jd_select],
        outputs=[resume_select, scope_hint, select_all_cb],
    )

    def sync_select_all(resume_ids):
        """
        用户手动增删简历后，同步「全选」勾选框状态。
        """

        all_ids = {
            item["doc_id"]
            for item in get_documents("resume")
            if item.get("doc_id")
        }

        picked = set(resume_ids or [])

        return gr.update(
            value=bool(all_ids) and picked == all_ids,
        )

    resume_select.change(
        sync_select_all,
        inputs=[resume_select],
        outputs=[select_all_cb],
    )

    # 一键清空岗位 / 简历的选择。
    clear_select_btn.click(
        lambda: (
            [],
            [],
            "**当前范围：** 未选择岗位 / 简历，"
            "Agent 将在整个知识库中检索。",
            False,
        ),
        outputs=[
            jd_select,
            resume_select,
            scope_hint,
            select_all_cb,
        ],
    )

    clear_btn.click(
        lambda: (
            [],
            "",
            [],
            [],
            "**当前范围：** 未选择岗位 / 简历，"
            "Agent 将在整个知识库中检索。",
            False,
        ),
        outputs=[
            chatbot,
            textbox,
            jd_select,
            resume_select,
            scope_hint,
            select_all_cb,
        ],
    )

    return (
        chatbot,
        textbox,
        jd_select,
        resume_select,
        scope_hint,
        refresh_btn,
        select_all_cb,
    )


# ============================================================
# 九、数据管理页
# ============================================================

def data_page() -> tuple:
    """
    数据管理页面。

    返回：

        (jd_card, resume_card, chunk_card, document_card,
         table, dropdown)
    """

    gr.HTML(
        _page_head(
            "Data",
            "数据管理",
            "查看知识库全貌，并清理不需要的文档。",
        )
    )

    with gr.Row():

        jd_card = gr.HTML(
            _stat_card("JD 岗位", "–", "已入库", "indigo")
        )

        resume_card = gr.HTML(
            _stat_card("候选人简历", "–", "已入库", "violet")
        )

        chunk_card = gr.HTML(
            _stat_card("向量片段", "–", "Chunks", "emerald")
        )

        document_card = gr.HTML(
            _stat_card("文档总数", "–", "JD + 简历", "amber")
        )

    gr.HTML(
        """
        <div class="section-title">
            全部文档
            <span class="hint">JD 与简历统一视图</span>
        </div>
        """
    )

    with gr.Column(elem_classes=["panel"]):

        table = gr.HTML(
            _doc_table([], "还没有任何文档")
        )

        with gr.Row():

            dropdown = gr.Dropdown(
                label="选择要删除的文档（可多选）",
                choices=[],
                value=[],
                multiselect=True,
                interactive=True,
                scale=4,
                elem_id="data-dropdown",
                elem_classes=["doc-select"],
            )

            refresh_btn = gr.Button(
                "↻  刷新",
                elem_classes=["ghost-btn"],
                scale=1,
            )

            delete_btn = gr.Button(
                "删除文档",
                elem_classes=["danger-btn"],
                scale=1,
            )

    def do_refresh():

        return refresh_documents("both")

    def do_delete(doc_ids):
        """
        批量删除选中的文档。
        """

        if not doc_ids:

            gr.Warning("请先选择要删除的文档。", duration=4)

            return refresh_documents("both")

        ok = 0
        failed = 0

        for doc_id in doc_ids:

            message = delete_document(doc_id)

            if message.startswith("✅"):

                ok += 1

            else:

                failed += 1

        if failed:

            gr.Warning(
                f"删除完成：成功 {ok} 个，失败 {failed} 个",
                duration=7,
            )

        else:

            gr.Info(f"已删除 {ok} 个文档", duration=4)

        return refresh_documents("both")

    refresh_btn.click(
        do_refresh,
        outputs=[table, dropdown],
    )

    delete_btn.click(
        do_delete,
        inputs=[dropdown],
        outputs=[table, dropdown],
    )

    return (
        jd_card,
        resume_card,
        chunk_card,
        document_card,
        table,
        dropdown,
    )

# ============================================================
# 十、系统设置页
# ============================================================

def settings_page():
    """
    系统设置页面。

    用户可以配置：

        1. DeepSeek API Key
        2. API Base URL
        3. Model

    配置保存到 data/config.json（已加入 .gitignore）。

    保存成功后会自动清理 LLM / Agent 缓存，
    让新配置立即生效。
    """

    gr.HTML(
        _page_head(
            "Settings",
            "系统设置",
            "配置 AI 服务。API Key 只保存在本地。",
        )
    )

    with gr.Column(elem_classes=["settings-card"]):

        gr.HTML(
            """
            <div class="section-title" style="margin-top:0;">
                AI 服务配置
            </div>

            <div class="settings-note">
                API Key 仅写入本机 data/config.json，
                不会上传到任何服务器。保存后会立即生效。
            </div>
            """
        )

        api_key = gr.Textbox(
            label="DeepSeek API Key",
            type="password",
            value=get_api_key(),
            placeholder="sk-xxxxxxxxxxxxxxxx",
        )

        base_url = gr.Textbox(
            label="API Base URL",
            value=get_base_url(),
            placeholder="https://api.deepseek.com",
        )

        model = gr.Textbox(
            label="Model",
            value=get_model(),
            placeholder="deepseek-chat",
        )

        with gr.Row():

            save_btn = gr.Button(
                "保存配置",
                elem_classes=["primary-btn"],
            )

            test_btn = gr.Button(
                "测试连接",
                elem_classes=["ghost-btn"],
            )

        status = gr.Markdown("")

    # --------------------------------------------------------
    # 本地 Embedding 说明
    # --------------------------------------------------------

    gr.HTML(
        _stat_card(
            "LOCAL EMBEDDING",
            "BAAI / BGE-M3",
            "本地运行，无需 API Key。",
            "emerald",
        )
    )

    # --------------------------------------------------------
    # 保存配置
    # --------------------------------------------------------

    def save_config(key, url, model_name):

        if not key.strip():

            return "❌ API Key 不能为空。"

        if not url.strip():

            return "❌ Base URL 不能为空。"

        if not model_name.strip():

            return "❌ Model 不能为空。"

        try:

            save_user_config(key, url, model_name)

        except Exception as exc:

            return (
                "### ❌ 保存失败\n\n"
                f"`{type(exc).__name__}`\n\n"
                f"{exc}"
            )

        # 配置改变后清缓存，让新 LLM / Agent 生效。
        clear_llm_cache()
        clear_agent_cache()

        return (
            "### ✓ 配置已保存\n\n"
            "新的 API 配置已经生效，"
            "现在可以进入 **AI Agent** 使用。"
        )

    # --------------------------------------------------------
    # 测试连接
    # --------------------------------------------------------

    def test_connection(key, url, model_name):

        if not key.strip():

            return "❌ API Key 不能为空。"

        if not url.strip():

            return "❌ Base URL 不能为空。"

        if not model_name.strip():

            return "❌ Model 不能为空。"

        try:

            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model_name.strip(),
                api_key=key.strip(),
                base_url=url.strip(),
                temperature=0,

                # 网络抖动时多等一会儿、多试几次
                timeout=60,
                max_retries=2,
            )

            response = llm.invoke(
                [
                    {
                        "role": "user",
                        "content": "请只回答：连接成功",
                    }
                ]
            )

            content = response.content

            if not isinstance(content, str):

                content = str(content)

            return (
                "### ✓ API 连接成功\n\n"
                f"**模型：** `{model_name}`\n\n"
                f"**返回：** `{content}`"
            )

        except Exception as exc:

            # 层层往下找真正的底层原因。
            #
            # OpenAI SDK 只会抛一句笼统的 "Connection error."，
            # 真正的原因（DNS / SSL / 超时 / 连接被拒）
            # 藏在 __cause__ / __context__ 里。
            root = exc

            for _ in range(6):

                nxt = (
                    getattr(root, "__cause__", None)
                    or getattr(root, "__context__", None)
                )

                if nxt is None or nxt is root:

                    break

                root = nxt

            detail = ""

            if root is not exc:

                detail = (
                    "\n\n**底层原因：** "
                    f"`{type(root).__name__}: {root}`"
                )

            return (
                "### ✕ API 连接失败\n\n"
                f"**错误类型：** `{type(exc).__name__}`\n\n"
                f"**错误信息：** `{exc}`"
                + detail
                + "\n\n---\n\n"
                "**排查建议：**\n\n"
                "1. 浏览器能否打开 https://api.deepseek.com\n"
                "2. Base URL 有没有填错（多空格 / 少 https://）\n"
                "3. 关掉 VPN / 代理后再试\n"
                "4. 稍等几秒再点一次测试"
            )

    save_btn.click(
        save_config,
        inputs=[api_key, base_url, model],
        outputs=[status],
    )

    test_btn.click(
        test_connection,
        inputs=[api_key, base_url, model],
        outputs=[status],
    )

# ============================================================
# 十一、主题
# ============================================================

def build_theme():
    """
    构建 Gradio 主题。

    Gradio 6 起 theme / css 需要在 launch() 时传入，
    因此这里单独暴露一个函数给 app.py 使用。
    """

    return gr.themes.Base(
        primary_hue="indigo",
        neutral_hue="slate",
    )


# ============================================================
# 十二、构建整个应用
# ============================================================

def build_app():
    """
    创建整个 Gradio 应用。

    整体结构：

        ┌────────────────────────────────────────────┐
        │ Sidebar              Main Content          │
        │                                            │
        │  概览                概览页                │
        │  JD 管理             JD 管理页             │
        │  简历库              简历库页              │
        │  AI Agent            Agent 对话页          │
        │  数据管理            数据管理页            │
        │  系统设置            系统设置页            │
        └────────────────────────────────────────────┘

    左侧点击导航不会刷新网页，
    而是通过 visible 切换右侧页面，
    因此整个项目是一个单页招聘工作台。
    """

    with gr.Blocks(
        title=f"{APP_TITLE} · {APP_SLOGAN}",
    ) as app:

        with gr.Row(elem_classes=["app-row"]):

            # =================================================
            # 左侧 Sidebar
            # =================================================

            with gr.Column(elem_classes=["sidebar"]):

                gr.HTML(
                    f"""
                    <div class="brand">
                        <div class="brand-logo">T</div>
                        <div>
                            <div class="brand-title">
                                {APP_TITLE}
                            </div>
                            <div class="brand-sub">
                                {APP_SLOGAN}
                            </div>
                        </div>
                    </div>
                    """
                )

                gr.HTML(
                    '<div class="nav-title">Workspace</div>'
                )

                nav_buttons = []

                for index, (label, icon) in enumerate(NAV_ITEMS):

                    classes = ["nav-btn"]

                    if index == PAGE_OVERVIEW:

                        classes.append("nav-active")

                    button = gr.Button(
                        f"{icon}   {label}",
                        elem_classes=classes,
                    )

                    nav_buttons.append(button)

                # ---------------------------------------------
                # 侧边栏状态卡
                # ---------------------------------------------

                gr.HTML(
                    """
                    <div class="side-status">
                        <div class="side-status-label">
                            SYSTEM STATUS
                        </div>
                        <div class="side-status-main">
                            <span class="dot"></span>
                            Agent Online
                        </div>
                        <div class="side-status-sub">
                            DeepSeek · BGE-M3<br>
                            Chroma · LangGraph
                        </div>
                    </div>
                    """
                )

            # =================================================
            # 右侧 Main Content
            # =================================================

            with gr.Column(elem_classes=["main"]):

                containers = []

                # ---------------------------------------------
                # 1. 概览
                # ---------------------------------------------

                with gr.Column() as overview_container:

                    (
                        dash_jd_card,
                        dash_resume_card,
                        dash_chunk_card,
                        dash_doc_card,
                        quick_jd_btn,
                        quick_resume_btn,
                        quick_agent_btn,
                    ) = overview_page()

                containers.append(overview_container)

                # ---------------------------------------------
                # 2. JD 管理
                # ---------------------------------------------

                with gr.Column(
                    visible=False
                ) as jd_container:

                    (
                        jd_table,
                        jd_dropdown,
                    ) = document_page(
                        prefix="jd",
                        doc_type="jd",
                        eyebrow="JD Library",
                        title="JD 管理",
                        desc="上传岗位描述，建立 JD 知识库。",
                        hint=(
                            "支持批量上传 PDF / DOCX，"
                            "也可以直接粘贴文本，"
                            "系统自动完成解析 → Chunk → 向量化 → 入库。"
                        ),
                    )

                containers.append(jd_container)

                # ---------------------------------------------
                # 3. 简历库
                # ---------------------------------------------

                with gr.Column(
                    visible=False
                ) as resume_container:

                    (
                        resume_table,
                        resume_dropdown,
                    ) = document_page(
                        prefix="resume",
                        doc_type="resume",
                        eyebrow="Resume Library",
                        title="简历库",
                        desc="上传候选人简历，建立候选人知识库。",
                        hint=(
                            "支持批量上传 PDF / DOCX，"
                            "也可以直接粘贴文本，"
                            "系统会自动建立向量索引。"
                        ),
                    )

                containers.append(resume_container)

                # ---------------------------------------------
                # 4. AI Agent
                # ---------------------------------------------

                with gr.Column(
                    visible=False
                ) as agent_container:

                    (
                        agent_chatbot,
                        agent_textbox,
                        agent_jd_select,
                        agent_resume_select,
                        agent_scope_hint,
                        agent_refresh_btn,
                        agent_select_all_cb,
                    ) = agent_page()

                containers.append(agent_container)

                # ---------------------------------------------
                # 5. 数据管理
                # ---------------------------------------------

                with gr.Column(
                    visible=False
                ) as data_container:

                    (
                        data_jd_card,
                        data_resume_card,
                        data_chunk_card,
                        data_doc_card,
                        data_table,
                        data_dropdown,
                    ) = data_page()

                containers.append(data_container)

                # ---------------------------------------------
                # 6. 系统设置
                # ---------------------------------------------

                with gr.Column(
                    visible=False
                ) as settings_container:

                    settings_page()

                containers.append(settings_container)

        # ====================================================
        # 页面切换
        # ====================================================

        nav_outputs = containers + nav_buttons

        def goto(index: int):
            """
            切换页面，并高亮对应的侧边导航。
            """

            container_updates = [
                gr.update(visible=(i == index))
                for i in range(len(containers))
            ]

            nav_updates = [
                gr.update(
                    elem_classes=(
                        ["nav-btn", "nav-active"]
                        if i == index
                        else ["nav-btn"]
                    )
                )
                for i in range(len(nav_buttons))
            ]

            return container_updates + nav_updates

        # 每个页面切换时顺便刷新自己的数据，
        # 这样用户看到的一定是最新结果。

        def open_overview():

            return goto(PAGE_OVERVIEW) + list(
                refresh_dashboard()
            )

        def open_jd():

            return goto(PAGE_JD) + list(
                refresh_documents("jd")
            )

        def open_resume():

            return goto(PAGE_RESUME) + list(
                refresh_documents("resume")
            )

        def open_agent(current_jd=None, current_resumes=None):

            return goto(PAGE_AGENT) + list(
                refresh_agent_selectors(current_jd, current_resumes)
            )

        def open_data():

            return (
                goto(PAGE_DATA)
                + list(refresh_dashboard())
                + list(refresh_documents("both"))
            )

        def open_settings():

            return goto(PAGE_SETTINGS)

        nav_buttons[PAGE_OVERVIEW].click(
            open_overview,
            outputs=nav_outputs + [
                dash_jd_card,
                dash_resume_card,
                dash_chunk_card,
                dash_doc_card,
            ],
        )

        nav_buttons[PAGE_JD].click(
            open_jd,
            outputs=nav_outputs + [
                jd_table,
                jd_dropdown,
            ],
        )

        nav_buttons[PAGE_RESUME].click(
            open_resume,
            outputs=nav_outputs + [
                resume_table,
                resume_dropdown,
            ],
        )

        nav_buttons[PAGE_AGENT].click(
            open_agent,
            inputs=[
                agent_jd_select,
                agent_resume_select,
            ],
            outputs=nav_outputs + [
                agent_jd_select,
                agent_resume_select,
                agent_scope_hint,
                agent_select_all_cb,
            ],
        )

        nav_buttons[PAGE_DATA].click(
            open_data,
            outputs=nav_outputs + [
                data_jd_card,
                data_resume_card,
                data_chunk_card,
                data_doc_card,
                data_table,
                data_dropdown,
            ],
        )

        nav_buttons[PAGE_SETTINGS].click(
            open_settings,
            outputs=nav_outputs,
        )

        # ====================================================
        # 概览页快速入口
        # ====================================================

        quick_jd_btn.click(
            open_jd,
            outputs=nav_outputs + [
                jd_table,
                jd_dropdown,
            ],
        )

        quick_resume_btn.click(
            open_resume,
            outputs=nav_outputs + [
                resume_table,
                resume_dropdown,
            ],
        )

        quick_agent_btn.click(
            open_agent,
            inputs=[
                agent_jd_select,
                agent_resume_select,
            ],
            outputs=nav_outputs + [
                agent_jd_select,
                agent_resume_select,
                agent_scope_hint,
                agent_select_all_cb,
            ],
        )

        # ====================================================
        # 启动时加载一次数据
        # ====================================================

        def bootstrap():

            return (
                *refresh_dashboard(),
                *refresh_documents("jd"),
                *refresh_documents("resume"),
                *refresh_dashboard(),
                *refresh_documents("both"),
                *refresh_agent_selectors(),
            )

        app.load(
            bootstrap,
            outputs=[
                dash_jd_card,
                dash_resume_card,
                dash_chunk_card,
                dash_doc_card,

                jd_table,
                jd_dropdown,

                resume_table,
                resume_dropdown,

                data_jd_card,
                data_resume_card,
                data_chunk_card,
                data_doc_card,

                data_table,
                data_dropdown,

                # AI Agent 的岗位 / 简历选择器
                agent_jd_select,
                agent_resume_select,
                agent_scope_hint,
                agent_select_all_cb,
            ],
        )

    return app