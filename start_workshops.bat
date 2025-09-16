@echo off
setlocal EnableDelayedExpansion
chcp 65001 > nul 2>&1

echo [INFO] Lancement des AI Workshops...

REM Arrêter les processus serveur existants
echo [INFO] Arrêt des processus serveur existants...
taskkill /f /im node.exe >nul 2>&1

REM Attendre un moment pour la libération des ports
timeout /t 2 >nul 2>&1

echo [INFO] Lancement du serveur HTTP pour les workshops...
echo [INFO] Ouverture sur http://localhost:8080/workshops_standalone.html
echo [INFO] Fonctionnalités workshops activées:
echo [INFO] - Interface dédiée aux ateliers pratiques
echo [INFO] - Bouton de thème 🎨 en haut à droite
echo [INFO] - Diagrammes Mermaid pour les flux de travail
echo [INFO] - Tableaux interactifs de sélection
echo [INFO] - Guide de structure des ateliers
echo.
echo [ASTUCE] Appuyez sur F pour le mode plein écran
echo [ASTUCE] Utilisez les flèches pour naviguer
echo [ASTUCE] ESC pour la vue d'ensemble
echo.

REM Lancer le serveur HTTP
start http://localhost:8080/workshops_standalone.html
npx http-server . -p 8080 --cors -o

echo.
echo [INFO] Serveur arrêté
pause