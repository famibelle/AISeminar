@echo off
echo [INFO] Arret des processus serveur existants...
taskkill /F /IM node.exe 2>nul

where node >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Installation/verification de live-server...
    npx live-server --version >nul 2>&1
    
    echo [INFO] Lancement du serveur avec live-reload...
    echo [INFO] Ouverture sur http://localhost:8080/index_standalone.html
    echo [INFO] Toutes les fonctionnalites activees:
    echo [INFO] - Bouton de theme 🌓 en haut a droite
    echo [INFO] - Diagrammes Mermaid integres
    echo [INFO] - Equations MathJax
    echo [INFO] - Mise en page optimisee
    echo [INFO] - LIVE RELOAD automatique des modifications !
    echo.
    echo [AVANTAGE] Les modifications du fichier .md sont automatiquement visibles
    echo.
    npx live-server --port=8080 --open=index_standalone.html --no-css-inject
) else (
    echo [ERREUR] Node.js est requis.
    pause
)