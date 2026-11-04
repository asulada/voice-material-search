@echo off
REM 启动后端（Windows）。只监听 127.0.0.1。
REM   run.bat                      默认库根目录 = 项目根\voice_library
REM   set VOICE_LIBRARY_ROOT=...  自定义库根目录
REM   set PORT=8000               自定义端口
chcp 65001 >nul
setlocal
cd /d "%~dp0backend" || (echo 找不到 backend 目录 & pause & exit /b 1)

if "%PY%"=="" set PY=py

rem 优先级：用户显式设置的 PY > 当前激活的 conda 环境 > virtualenv > py 默认。
rem 注意：Windows 的 py launcher 只认 VIRTUAL_ENV，**不认 CONDA_PREFIX**，
rem 所以 conda 环境下必须显式指向 %CONDA_PREFIX%\python.exe，否则会跑到 base 环境。
if "%PY%"=="py" if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" set PY=%CONDA_PREFIX%\python.exe

if "%PORT%"=="" set PORT=8000

rem 让 Python 的输入输出也用 UTF-8，避免 uvicorn 中文日志报 UnicodeEncodeError
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

"%PY%" --version >nul 2>&1
if errorlevel 1 (
  echo 找不到 Python 命令 "%PY%"，请安装 Python 或设置 PY 环境变量
  pause & exit /b 1
)

"%PY%" -c "import fastapi" 2>nul
if errorlevel 1 (
  echo 缺少依赖。先执行： %PY% -m pip install -r requirements.txt
  pause & exit /b 1
)

"%PY%" -m uvicorn server:app --host 127.0.0.1 --port %PORT% %*
pause