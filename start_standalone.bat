@echo off
echo [INFO] Arret des processus serveur existants...
taskkill /F /IM node.exe 2>nul

where node >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [INFO] Lancement du serveur HTTP avec auto-reload...
    echo [INFO] Ouverture sur http://localhost:8080/index_standalone.html
    echo [INFO] Toutes les fonctionnalites activees:
    echo [INFO] - Bouton de theme 🌓 en haut a droite
    echo [INFO] - Diagrammes Mermaid integres
    echo [INFO] - Equations MathJax
    echo [INFO] - Mise en page optimisee
    echo [INFO] - Auto-reload des modifications (F5 pour recharger)
    echo.
    echo [ASTUCE] Apres modification du fichier .md, appuyez sur F5 dans le navigateur
    echo.
    npx http-server -p 8080 -o index_standalone.html --cors
) else (
    echo [ERREUR] Node.js est requis.
    pause
)