@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul 2>&1

echo [INFO] Lancement de YOLO Reveal Auto...

REM Vérifier si l'environnement virtuel existe à la racine
if not exist ".venv" (
    echo [INFO] Création de l'environnement virtuel Python à la racine...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [ERREUR] Impossible de créer l'environnement virtuel
        echo [INFO] Vérifiez que Python est installé et accessible
        pause
        exit /b 1
    )
    echo [INFO] Environnement virtuel créé avec succès
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

REM Vérifier si requirements.txt existe dans Labs et installer les dépendances
if exist "requirements.txt" (
    echo [INFO] Installation des dépendances depuis Labs/requirements.txt...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo [ATTENTION] Erreur lors de l'installation des dépendances
        echo [INFO] Tentative de continuation...
    )
) else (
    echo [INFO] Aucun fichier requirements.txt trouvé dans Labs
    echo [INFO] Installation des dépendances principales...
    pip install opencv-python ultralytics pillow numpy
)

REM Vérifier si le script Python existe
if not exist "yolo_reveal_auto.py" (
    echo [ERREUR] Le fichier yolo_reveal_auto.py n'existe pas dans ce répertoire
    echo [INFO] Répertoire actuel: %CD%
    pause
    exit /b 1
)

echo [INFO] Lancement du script YOLO...
echo [INFO] Pour arrêter le script, utilisez Ctrl+C
echo ================================================

REM Lancer le script Python
python yolo_reveal_auto.py

REM Capturer le code de sortie
set SCRIPT_EXIT_CODE=!errorlevel!

echo ================================================
if !SCRIPT_EXIT_CODE! equ 0 (
    echo [INFO] Script terminé avec succès
) else (
    echo [ATTENTION] Script terminé avec le code d'erreur: !SCRIPT_EXIT_CODE!
)

echo [INFO] Appuyez sur une touche pour fermer cette fenêtre...
pause > nul