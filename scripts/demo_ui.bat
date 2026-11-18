@echo off
REM Serve the PathSnap UI on Windows (Laptop 2).  Usage: scripts\demo_ui.bat http://10.20.201.72:8000
cd /d "%~dp0..\ui"
set API=%1
if "%API%"=="" set API=http://localhost:8000
echo PathSnap UI ^| http://localhost:5500/?api=%API%
start "" "http://localhost:5500/?api=%API%"
python -m http.server 5500 --bind 127.0.0.1
