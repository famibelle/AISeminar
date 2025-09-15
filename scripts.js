// Scripts personnalisés pour AI Seminar
// Basé sur les fonctionnalités de SourceCode\reveal.js\index.html

// Stockage global pour les définitions Mermaid originales
let mermaidDefinitions = new Map();

// Fonction pour ajuster dynamiquement la taille du texte pour qu'il s'adapte au slide
function adjustFontSize() {
  document.querySelectorAll('.slides section').forEach(slide => {
    let fontSize = 100; // Commence à 100%
    slide.style.fontSize = fontSize + '%';
    
    // Réduit la taille jusqu'à ce que le contenu s'ajuste complètement
    while (slide.scrollHeight > slide.clientHeight && fontSize > 10) {
      fontSize -= 2;
      slide.style.fontSize = fontSize + '%';
    }
  });
}

// Fonction pour basculer entre les thèmes
function toggleTheme() {
  const reveal = document.querySelector('.reveal');
  const isLightTheme = reveal.classList.contains('light-theme');
  
  if (isLightTheme) {
    // Passer au mode sombre
    reveal.classList.remove('light-theme');
    localStorage.setItem('reveal-theme', 'dark');
    updateMermaidTheme('dark');
  } else {
    // Passer au mode clair
    reveal.classList.add('light-theme');
    localStorage.setItem('reveal-theme', 'light');
    updateMermaidTheme('light');
  }
}

// Fonction pour mettre à jour le thème Mermaid
function updateMermaidTheme(theme) {
  // Ne pas essayer de re-rendre si aucun diagramme n'existe
  const mermaidElements = document.querySelectorAll('.mermaid');
  if (mermaidElements.length === 0) return;
  
  // Configuration du thème
  const mermaidConfig = {
    startOnLoad: false,
    logLevel: 'error', // Réduire les logs pour éviter le spam
    securityLevel: 'loose',
    flowchart: { 
      htmlLabels: false,
      useMaxWidth: true
    }
  };
  
  if (theme === 'light') {
    mermaidConfig.theme = 'default';
    mermaidConfig.themeVariables = {
      primaryColor: '#333333',
      primaryTextColor: '#333333',
      primaryBorderColor: '#333333',
      lineColor: '#333333'
    };
  } else {
    mermaidConfig.theme = 'dark';
    mermaidConfig.themeVariables = {
      primaryColor: '#ffffff',
      primaryTextColor: '#ffffff',
      primaryBorderColor: '#ffffff',
      lineColor: '#ffffff'
    };
  }
  
  // Réinitialiser Mermaid avec la nouvelle configuration
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize(mermaidConfig);
    
    // Re-rendre chaque diagramme individuellement
    mermaidElements.forEach((element, index) => {
      const originalDefinition = mermaidDefinitions.get(element.id);
      if (originalDefinition) {
        // Nettoyer l'élément
        element.innerHTML = '';
        element.removeAttribute('data-processed');
        
        // Re-créer le contenu
        element.textContent = originalDefinition;
        
        // Re-rendre avec un délai pour éviter les conflits
        setTimeout(() => {
          try {
            mermaid.init(undefined, `#${element.id}`);
          } catch (error) {
            console.warn('Erreur lors du re-rendu Mermaid:', error);
            // En cas d'erreur, restaurer le texte original
            element.textContent = originalDefinition;
          }
        }, index * 50); // Délai progressif pour chaque diagramme
      }
    });
    
    // Re-rendre MathJax après le changement de thème
    setTimeout(() => {
      if (window.MathJax) {
        MathJax.typesetPromise();
      }
    }, (mermaidElements.length * 50) + 200);
  }
}

// Restaurer le thème sauvegardé au chargement
function restoreTheme() {
  const savedTheme = localStorage.getItem('reveal-theme') || 'dark';
  const reveal = document.querySelector('.reveal');
  
  if (savedTheme === 'light') {
    reveal.classList.add('light-theme');
  }
}

// Fonction pour traiter les diagrammes Mermaid
function processMermaidDiagrams() {
  // Cherche différents sélecteurs possibles pour les blocs Mermaid
  const selectors = [
    'code.language-mermaid',
    'code[class*="mermaid"]',
    'pre code.language-mermaid',
    'pre code[class*="mermaid"]'
  ];
  
  let processed = 0;
  selectors.forEach(selector => {
    document.querySelectorAll(selector).forEach((element, index) => {
      const graphDefinition = element.textContent.trim();
      if (graphDefinition && !element.dataset.processed) {
        const graphDiv = document.createElement('div');
        graphDiv.className = 'mermaid';
        graphDiv.id = 'mermaid-' + processed;
        graphDiv.textContent = graphDefinition;
        
        // Stocker la définition originale pour le changement de thème
        mermaidDefinitions.set(graphDiv.id, graphDefinition);
        
        // Marque l'élément comme traité
        element.dataset.processed = 'true';
        
        // Remplace le code par le div mermaid
        const parentPre = element.closest('pre');
        if (parentPre) {
          parentPre.parentNode.replaceChild(graphDiv, parentPre);
        } else {
          element.parentNode.replaceChild(graphDiv, element);
        }
        processed++;
      }
    });
  });
  
  // Initialise Mermaid après avoir créé les divs
  if (processed > 0 && typeof mermaid !== 'undefined') {
    setTimeout(() => {
      try {
        mermaid.init(undefined, '.mermaid');
      } catch (error) {
        console.warn('Erreur lors de l\'initialisation Mermaid:', error);
      }
    }, 100);
  }
}

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', function() {
  console.log('Scripts.js: DOMContentLoaded');
  
  // Restaurer le thème
  restoreTheme();
  
  // Ajouter la classe smaller à toutes les sections
  setTimeout(() => {
    document.querySelectorAll('.slides section').forEach(slide => {
      slide.classList.add('smaller');
    });
    adjustFontSize();
  }, 100);
});

// Ajustement lors du redimensionnement
window.addEventListener('resize', adjustFontSize);

// Configuration pour Reveal.js - s'exécute quand Reveal est disponible
function initRevealFeatures() {
  console.log('Initialisation des fonctionnalités Reveal...');
  
  // Configuration de Mermaid
  if (typeof mermaid !== 'undefined') {
    mermaid.initialize({
      startOnLoad: false, // Désactivé pour éviter les conflits
      theme: 'dark', // Correspond au thème noir de Reveal.js
      logLevel: 'error',
      securityLevel: 'loose', // Permet plus de flexibilité
      flowchart: { 
        htmlLabels: false,
        useMaxWidth: true
      },
      themeVariables: {
        primaryColor: '#ffffff',
        primaryTextColor: '#ffffff',
        primaryBorderColor: '#ffffff',
        lineColor: '#ffffff'
      }
    });
  }
  
  // Événements Reveal.js
  if (typeof Reveal !== 'undefined') {
    Reveal.addEventListener('ready', function() {
      console.log('Reveal ready event');
      setTimeout(() => {
        processMermaidDiagrams();
        adjustFontSize();
        // Re-rendre MathJax après le chargement complet
        if (window.MathJax) {
          MathJax.typesetPromise();
        }
      }, 500);
    });
    
    Reveal.addEventListener('slidechanged', function() {
      setTimeout(() => {
        adjustFontSize();
        // Re-rendre MathJax lors du changement de slide
        if (window.MathJax) {
          MathJax.typesetPromise();
        }
      }, 100);
    });
  }
}

// Attendre que Reveal soit disponible
function waitForReveal() {
  if (typeof Reveal !== 'undefined') {
    initRevealFeatures();
  } else {
    setTimeout(waitForReveal, 100);
  }
}

// Démarrer l'attente pour Reveal
setTimeout(waitForReveal, 100);