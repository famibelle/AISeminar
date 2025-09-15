@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul 2>&1

echo [INFO] Lancement rapide de YOLO Reveal Auto...

REM Vérifier si l'environnement virtuel existe à la racine
if not exist ".venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel non trouvé à la racine
    echo [INFO] Lancez d'abord start_yolo_reveal.bat pour créer l'environnement
    pause
    exit /b 1
)

REM Activer l'environnement virtuel de la racine
echo [INFO] Activation de l'environnement virtuel...
call .venv\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo [ERREUR] Impossible d'activer l'environnement virtuel
    pause
    exit /b 1
)

REM Changer vers le répertoire Labs après activation de l'env virtuel
echo [INFO] Navigation vers le répertoire Labs...
cd /d "%~dp0Labs"

REM Lancer le script Python
echo [INFO] Lancement du script YOLO...
python yolo_reveal_auto.py

echo [INFO] Appuyez sur une touche pour fermer...
pause > nul