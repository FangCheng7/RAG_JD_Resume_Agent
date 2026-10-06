@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================
echo   TalentMatch AI - 打包成 exe
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 没找到 .venv，请先在项目目录执行：
    echo        python -m venv .venv
    echo        .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

echo [1/3] 安装 PyInstaller ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check pyinstaller
if errorlevel 1 (
    echo [错误] PyInstaller 安装失败
    pause
    exit /b 1
)

echo.
echo [2/3] 开始打包（首次约 5-15 分钟，请耐心等待）...
".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean TalentMatchAI.spec
if errorlevel 1 (
    echo [错误] 打包失败，请查看上面的报错
    pause
    exit /b 1
)

echo.
echo [3/3] 打包完成！
echo.
echo   输出目录：dist\TalentMatchAI\
echo   可执行文件：dist\TalentMatchAI\TalentMatchAI.exe
echo.
echo   把整个 TalentMatchAI 文件夹压缩后发给别人，
echo   对方解压双击 TalentMatchAI.exe 即可使用。
echo   （首次运行需要联网下载 BGE-M3 模型，约 2.3GB）
echo.
pause