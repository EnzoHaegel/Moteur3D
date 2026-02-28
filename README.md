# Moteur 3D

Moteur de rendu 3D accéléré GPU écrit en Python avec OpenGL 3.3, NumPy et Pygame. Il propose une caméra FPS style Minecraft (ZQSD + souris), un chargeur de modèles OBJ, et un pipeline de rendu GPU avec shaders et éclairage directionnel.

## Setup

**Prérequis** : Python 3.10+

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Lancer le moteur

```bash
python main.py
```

Le moteur charge automatiquement le modèle `model/FinalBaseMesh.obj` s'il est présent, sinon il démarre avec une scène vide (grille uniquement).

## Contrôles

| Touche | Action |
|--------|--------|
| Z / W | Avancer |
| S | Reculer |
| Q / A | Aller à gauche |
| D | Aller à droite |
| Espace | Monter |
| Shift | Descendre |
| Souris | Regarder autour |
| F1 | Basculer solide / fil de fer |
| F2 | Afficher / masquer la grille |
| F3 | Afficher / masquer le HUD |
| Échap | Libérer la souris / Quitter |

## Lancer les tests

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
```

Avec coverage :

```bash
python -m pytest tests/ --cov=engine --cov=main --cov-report=term-missing
```

Ou directement depuis VS Code via l'onglet **Testing** (configuration pytest incluse dans `.vscode/settings.json`).

## Structure du projet

```
├── main.py              # Point d'entrée
├── requirements.txt     # Dépendances (numpy, pygame, PyOpenGL)
├── .gitignore           # Fichiers ignorés par Git
├── engine/
│   ├── math3d.py        # Vec3, Mat4 — maths 3D optimisées NumPy
│   ├── camera.py        # Caméra FPS avec cache des matrices
│   ├── mesh.py          # Mesh + chargeur OBJ avec triangulation
│   ├── renderer.py      # Pipeline de rendu OpenGL (shaders, VAO/VBO)
│   └── engine.py        # Boucle principale, gestion des entrées
├── tests/
│   ├── test_math3d.py   # Tests Vec3 et Mat4
│   ├── test_camera.py   # Tests caméra (déplacement, souris, cache)
│   ├── test_mesh.py     # Tests Mesh et OBJLoader
│   ├── test_engine.py   # Tests Engine (init, events, rendu, boucle)
│   ├── test_renderer.py # Tests Renderer OpenGL (shaders, upload, draw)
│   └── test_main.py     # Tests point d'entrée
└── model/
    └── FinalBaseMesh.obj
```

## Optimisations

- Rendu GPU via OpenGL 3.3 core (vertex/fragment shaders GLSL)
- VSync activée (FPS synchronisé au taux de rafraîchissement de l'écran)
- Géométrie uploadée une seule fois en VRAM (VAO/VBO/EBO statiques)
- Back-face culling, near/far clipping et depth test gérés par le GPU
- HUD rendu sur une surface Pygame, projeté en overlay texture OpenGL
- Cache des matrices vue et projection (recalcul uniquement si modifiées)
- `__slots__` sur toutes les classes pour réduire l'empreinte mémoire
- PyOpenGL-accelerate pour des appels GL rapides