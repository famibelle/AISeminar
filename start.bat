@echo off
setlocal

:: Vérifie si un argument est passé
if "%~1"=="" goto menu

:: Si l'argument est "1" ou "dynamic", lance le mode dynamique
if "%~1"=="1" goto dynamic_mode
if /I "%~1"=="dynamic" goto dynamic_mode

:: Si l'argument est "2" ou "static", lance le mode statique
if "%~1"=="2" goto static_mode
if /I "%~1"=="static" goto static_mode

:: Sinon, affiche le menu
:menu
cls
echo 1 - Dynamic mode (local server, auto-refresh)
echo 2 - Static mode (generate standalone HTML)
echo 3 - Standalone mode (all features: theme toggle, mermaid, etc.)
echo.
set /p choice=Choose a mode (1, 2, or 3):

if "%choice%"=="1" goto dynamic_mode
if "%choice%"=="2" goto static_mode
if "%choice%"=="3" goto standalone_mode

echo Invalid choice. Please enter 1, 2, or 3.
pause
exit /b

:dynamic_mode
where node >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo Launching dynamic mode...
    npx reveal-md ai_seminar_slides.md --watch
) else (
    echo Node.js is not installed.
    set /p download=Download Node.js now (Y/N)?
    if /I "%download%"=="Y" start "" "https://nodejs.org/"
)
pause
exit /b

:static_mode
echo Generating static version...
npx reveal-md ai_seminar_slides.md --static _docs
start "" "_docs\index.html"
pause
exit /b

:standalone_mode
echo [INFO] Launching standalone mode with all features...
echo [INFO] - Theme toggle button 🌓
echo [INFO] - Mermaid diagrams
echo [INFO] - MathJax equations
echo [INFO] - Optimized layout
call start_standalone.bat
exit /b
