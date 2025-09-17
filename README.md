# 🎯 AI Seminar for Automotive Experts

Une présentation interactive sur l'Intelligence Artificielle appliquée à l'industrie automobile, incluant des démonstrations en temps réel avec YOLO, des diagrammes Mermaid, et des équations MathJax.

## 📋 Table des matières

- [🚀 Démarrage rapide](#-démarrage-rapide)
- [📖 Contenu de la présentation](#-contenu-de-la-présentation)
- [🔧 Prérequis](#-prérequis)
- [💻 Installation](#-installation)
- [🎪 Utilisation](#-utilisation)
- [🛠️ Scripts disponibles](#️-scripts-disponibles)
- [🎨 Fonctionnalités](#-fonctionnalités)
- [🐛 Dépannage](#-dépannage)
- [📁 Structure du projet](#-structure-du-projet)

## 🚀 Démarrage rapide

### Lancement de la présentation

```bash
# Cloner le projet
git clone https://github.com/famibelle/AISeminar.git
cd AISeminar

# Lancer la présentation autonome
.\start_standalone.bat
```

La présentation s'ouvrira automatiquement dans votre navigateur à l'adresse : **http://127.0.0.1:8080/index.html**

### Accès aux workshops pratiques

```bash
# Lancer l'interface des ateliers
.\start_workshops.bat
```

Interface workshops accessible à : **http://127.0.0.1:8080/workshops_standalone.html**

## 📖 Contenu de la présentation

**Durée** : 2 heures (14:00-16:00)

### Programme détaillé :
- **14:00-14:30** : Introduction à l'IA & concepts fondamentaux
- **14:30-14:50** : Natural Language Processing (NLP) & Large Language Models (LLM)
- **14:50-15:10** : Computer Vision et Multimodalité
- **15:10-15:30** : Data, Documentation Technique et Code Legacy
- **15:30-15:50** : IA appliquée à l'industrie automobile
- **15:50-16:00** : Résumé

### Démonstrations incluses :
- 🎯 **YOLO en temps réel** : Détection d'objets à 27 FPS
- 📊 **Diagrammes Mermaid** : Visualisations interactives
- 🧮 **Équations MathJax** : Formules mathématiques
- 🎨 **Thèmes multiples** : Mode sombre/clair

## 🛠️ Ateliers pratiques (Partie 2)

**Durée** : 1h30 • **Groupes** : 3-5 participants

### Ateliers disponibles :
- 🔍 **RAG et Documentation Technique** (STT + recherche sémantique)
- 💻 **Code Legacy** : Documentation, maintenance, transposition
- ⚙️ **Conception Mécanique Assistée par IA**
- 📊 **Spécifications Logicielles et Matrices de Déviation**
- 🌡️ **Capteurs Virtuels et Estimation Indirecte** (température, pression, couple)
- 🚗 **Surveillance Thermique et Vehicle Dynamics Control**
- 🧪 **Tests** : Nouveaux scénarios, réduction des tests
- 🔬 **Simulation** : Complémenter les activités de simulation par l'IA

**Format** : Objectif concret • Solution pratique • Débriefing collectif

## 🔧 Prérequis

### Système requis :
- **Windows 10/11** avec PowerShell
- **Node.js** (version 14+ recommandée)
- **Navigateur moderne** (Chrome, Firefox, Edge)

### Optionnel (pour YOLO) :
- **Python 3.8+**
- **OpenCV, Ultralytics**
- **Webcam** pour les démonstrations live

## 💻 Installation

### 1. Installation des dépendances Node.js

```bash
# Installer http-server pour servir la présentation
npm install
```

### 2. Configuration Python (optionnel - pour YOLO)

```bash
# Première utilisation - configuration complète
.\start_yolo_reveal.bat

# Utilisations suivantes - lancement rapide
.\start_yolo_quick.bat
```

## 🎪 Utilisation

### Mode présentation principal

```bash
# Lancement autonome avec toutes les fonctionnalités
.\start_standalone.bat
```

### Mode workshops pratiques

```bash
# Interface dédiée aux ateliers pratiques
.\start_workshops.bat
```

**Fonctionnalités activées :**
- ✅ Serveur HTTP sur port 8080
- ✅ CORS activé pour les ressources locales
- ✅ Bouton de changement de thème (🎨 en haut à droite)
- ✅ Diagrammes Mermaid intégrés
- ✅ Équations MathJax
- ✅ Mise en page optimisée
- ✅ Support des iframes (YOLO)

### Mode avec rechargement automatique

```bash
# Auto-reload des modifications (développement)
.\start_standalone_live.bat
```

## 🛠️ Scripts disponibles

| Script | Description | Usage |
|--------|-------------|-------|
| `start_standalone.bat` | **Principal** - Présentation autonome | Production |
| `start_workshops.bat` | **Workshops** - Interface ateliers pratiques | Ateliers |
| `start_standalone_live.bat` | Avec auto-reload | Développement |
| `start_yolo_reveal.bat` | Configuration YOLO complète | Première fois |
| `start_yolo_quick.bat` | Lancement YOLO rapide | Usage répété |
| `test_404.bat` | Test des ressources | Diagnostic |

## 🎨 Fonctionnalités

### Interface de présentation
- 🎯 **Navigation** : Flèches directionnelles, espace, ESC
- 🎨 **Thèmes** : Bouton de basculement sombre/clair
- 📱 **Responsive** : Adaptation automatique à la taille d'écran
- ⌨️ **Raccourcis** : 
  - `F` : Mode plein écran
  - `ESC` : Vue d'ensemble
  - `S` : Notes du présentateur

### Démonstrations techniques
- 🎯 **YOLO Live** : Détection d'objets en temps réel
- 📊 **Mermaid** : Diagrammes de flux, ER, Gitgraph
- 🧮 **MathJax** : Équations LaTeX intégrées
- 🎥 **Médias** : Vidéos et images optimisées

### Optimisations
- ⚡ **Performance** : Chargement optimisé des ressources
- 🌐 **CDN** : Reveal.js, Mermaid, MathJax via CDN
- 🔒 **Sécurité** : Configuration CORS appropriée
- 📦 **Standalone** : Fonctionne sans connexion internet (sauf CDN)
- 🔧 **Modulaire** : Contenu markdown séparé pour faciliter la maintenance

### 📝 Avantages de la structure modulaire

Les workshops utilisent désormais une approche modulaire :
- `workshops_slides.md` : Contenu des slides uniquement
- `workshops_standalone.html` : Interface et configuration
- Facilite la modification du contenu sans toucher au code HTML
- Permet la réutilisation du contenu dans d'autres contextes

## 🐛 Dépannage

### Problèmes courants

#### Erreur de port occupé
```bash
# Tuer les processus existants
taskkill /f /im node.exe
.\start_standalone.bat
```

#### Problèmes de permissions
```bash
# Exécuter en tant qu'administrateur
# Clic droit > "Exécuter en tant qu'administrateur"
```

#### YOLO ne se lance pas
```bash
# Reconfigurer l'environnement Python
.\start_yolo_reveal.bat
```

#### Diagrammes Mermaid ne s'affichent pas
- Vérifiez la console (F12) pour les erreurs
- Rechargez la page (F5)
- Basculez le thème (bouton 🎨)

### Logs et diagnostic

```bash
# Tester l'accessibilité des ressources
.\test_404.bat

# Vérifier les logs du serveur dans le terminal
# Les codes 200 = OK, 404 = Fichier non trouvé
```

## 📁 Structure du projet

```
AISeminar/
├── 📄 README.md                    # Ce fichier
├── 🎯 ai_seminar_slides.md         # Contenu de la présentation
├── 🛠️ workshops.md                 # Contenu original des ateliers
├── 📝 workshops_slides.md          # Slides workshops (modulaire)
├── 🌐 index.html                   # Interface principale
├── 🎪 workshops_standalone.html    # Interface workshops (charge workshops_slides.md)
├── 🎨 favicon.svg                  # Icône du site
├── 🚀 start_standalone.bat         # Script de lancement principal
├── 🛠️ start_workshops.bat          # Script de lancement workshops
├── ⚡ start_standalone_live.bat    # Version avec auto-reload
├── 📊 IMGs/                        # Images et médias
├── 🔬 Labs/                        # Laboratoires et démos
│   ├── 🎯 yolo_live_stream.html    # Démo YOLO en temps réel
│   ├── 🐍 yolo_reveal_auto.py      # Script Python YOLO
│   ├── 📚 *.ipynb                  # Notebooks Jupyter
│   └── 📋 requirements.txt         # Dépendances Python
├── 🛠️ scripts.js                   # Scripts personnalisés
├── 🎨 styles.css                   # Styles personnalisés
└── 📦 package.json                 # Dépendances Node.js
```

## 📞 Support

### Contacts
- **Auteur** : Médhi Famibelle
- **Email** : [Contact via GitHub](https://github.com/famibelle)
- **Repository** : [AISeminar](https://github.com/famibelle/AISeminar)

### Ressources
- 📖 [Documentation Reveal.js](https://revealjs.com/)
- 📊 [Syntaxe Mermaid](https://mermaid.js.org/)
- 🧮 [Guide MathJax](https://docs.mathjax.org/)
- 🎯 [Documentation YOLO](https://docs.ultralytics.com/)

---

## 🏁 Commencer maintenant

```bash
# Une seule commande pour tout lancer !
.\start_standalone.bat
```

**Bonne présentation ! 🚀**