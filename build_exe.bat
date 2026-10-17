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

echo [1/5] 安装 PyInstaller ...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check pyinstaller
if errorlevel 1 (
    echo [错误] PyInstaller 安装失败
    pause
    exit /b 1
)

echo.
echo [2/5] 开始打包（首次约 5-15 分钟，请耐心等待）...

rem ------------------------------------------------------------
rem 默认输出到项目下的 dist\TalentMatchAI\
rem C 盘空间紧张的话，把下面两行改成 E 盘，例如：
rem     set DIST=E:\TalentMatchAI_dist
rem     set WORK=E:\TalentMatchAI_build
rem ------------------------------------------------------------
set DIST=dist
set WORK=build

".venv\Scripts\python.exe" -m PyInstaller --noconfirm --clean --distpath "%DIST%" --workpath "%WORK%" TalentMatchAI.spec
if errorlevel 1 (
    echo [错误] 打包失败，请查看上面的报错
    pause
    exit /b 1
)

echo.
echo [3/5] 复制使用说明到输出目录 ...
if exist "启动说明.txt" (
    copy /y "启动说明.txt" "%DIST%\TalentMatchAI\启动说明.txt" >nul
    echo   已复制 启动说明.txt
) else (
    echo   [跳过] 没找到 启动说明.txt
)

echo.
echo [4/5] 复制 Embedding 模型到输出目录 ...
rem 把模型一起打进分发包，别人下载后就不用再下模型了。
if exist "models" (
    robocopy "models" "%DIST%\TalentMatchAI\models" /E /NFL /NDL /NJH /NJS /NP >nul
    echo   已复制 models\ 文件夹
) else (
    echo   [警告] 没找到 models 文件夹！
    echo          请先运行: .venv\Scripts\python.exe download_model.py
    echo          否则别人用的时候会联网下载模型。
)

echo.
echo [5/5] 打包完成！
echo.
echo   输出目录：%DIST%\TalentMatchAI\
echo   可执行文件：%DIST%\TalentMatchAI\TalentMatchAI.exe
echo.
echo   把整个 TalentMatchAI 文件夹压缩成 zip 后发给别人，
echo   对方解压双击 TalentMatchAI.exe 即可使用。
echo.
echo   首次运行需要联网下载 BGE-M3 模型（约 2.3GB）；
echo   也可以把已下好的 models\bge-m3 放到 exe 同级目录跳过。
echo.
pause