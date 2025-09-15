@echo off
echo [INFO] Arret des processus reveal-md existants...
taskkill /F /IM node.exe 2>nul

where node >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Lancement en mode dynamique avec configuration personnalisee...
    echo [INFO] Utilisation des fichiers: reveal-md.config.yml, styles.css, scripts.js
    echo [INFO] Ouverture sur http://localhost:1950/ai_seminar_slides.md
    echo [INFO] Bouton de theme 🌓 disponible en haut a droite
    npx reveal-md ai_seminar_slides.md --config reveal-md.config.yml --port 1950 --watch
) else (
    echo [ERREUR] Node.js est requis.
    pause
)
