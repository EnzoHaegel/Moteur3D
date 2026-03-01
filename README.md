# Moteur 3D

Moteur de rendu 3D accéléré GPU écrit en Python avec OpenGL 3.3, NumPy et Pygame. Il propose une caméra FPS (ZQSD + souris), un chargeur de modèles OBJ, un pipeline de rendu GPU avec shaders et éclairage directionnel, ainsi qu'une API de simulation pour le reinforcement learning.

## Setup

**Prérequis** : Python 3.10+

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Lancer le moteur

```bash
python main.py
```

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

## Scene Graph

```python
from engine import Engine, Primitives, Vec3

engine = Engine()

cube = Primitives.cube(size=2.0)
engine.add_object("red_cube", cube,
    position=Vec3(0, 1, 0), color=(1, 0, 0))

robot = engine.get_object("red_cube")
robot.transform.position = Vec3(5, 1, 0)
engine.remove_object(robot)
```

## Primitives procédurales

```python
from engine import Primitives

cube     = Primitives.cube(size=1.0)
sphere   = Primitives.sphere(radius=0.5, segments=16, rings=16)
cylinder = Primitives.cylinder(radius=0.5, height=2.0, segments=16)
plane    = Primitives.plane(width=10.0, depth=10.0)
```

## Physique

Le moteur intègre un moteur physique complet avec timestep fixe, gravité, collisions et résolution par impulsions.

### Matériaux

Chaque objet a un matériau qui définit son comportement physique :

```python
from engine import PhysicsMaterial

# Presets disponibles
PhysicsMaterial.STONE    # friction=0.6, rebond=0.1, densité=2500
PhysicsMaterial.RUBBER   # friction=0.9, rebond=0.8, densité=1100
PhysicsMaterial.ICE      # friction=0.05, rebond=0.1, densité=917
PhysicsMaterial.WOOD     # friction=0.4, rebond=0.3, densité=600
PhysicsMaterial.METAL    # friction=0.3, rebond=0.2, densité=7800

# Matériau personnalisé
mat = PhysicsMaterial(friction=0.7, restitution=0.5, density=1200.0)
```

### Rigid Bodies et Gravité

```python
engine = Engine()

# Sol statique (mass=0 ou static=True)
engine.add_object("ground", Primitives.plane(width=50, depth=50),
    material=PhysicsMaterial.STONE, static=True)

# Balle qui tombe (affectée par la gravité)
engine.add_object("ball", Primitives.sphere(radius=0.5),
    position=Vec3(0, 10, 0), mass=1.0,
    material=PhysicsMaterial.RUBBER, color=(1, 0, 0))

# Modifier la gravité
engine.physics.gravity.acceleration = Vec3(0, -3.7, 0)  # Mars
```

### Collisions AABB et Raycasting

```python
from engine import AABB, Ray, ray_aabb_intersect, Vec3

obj_a = engine.get_object("cube_a")
obj_b = engine.get_object("cube_b")

if obj_a.get_aabb().intersects(obj_b.get_aabb()):
    print("Collision !")

ray = Ray(Vec3(0, 1, -5), Vec3(0, 0, 1))
hit = ray_aabb_intersect(ray, obj_a.get_aabb())
if hit is not None:
    print(f"Touché à distance {hit:.2f}")
```

### Joints articulés (pour robots RL)

```python
torso = engine.add_object("torso", Primitives.cube(size=0.5),
    position=Vec3(0, 2, 0), color=(0, 1, 0))
thigh = engine.add_object("thigh", Primitives.cylinder(radius=0.1, height=0.5),
    position=Vec3(0, 1.5, 0), color=(0, 0.8, 0))

# Joint charnière (genou, coude)
hip = engine.physics.add_hinge_joint(torso, thigh,
    anchor_a=Vec3(0, -0.25, 0), axis=Vec3(1, 0, 0),
    min_angle=-90, max_angle=90, motor_max_force=50.0)

# Contrôle moteur par l'agent RL
hip.motor_enabled = True
hip.motor_speed = 45.0  # degrés/seconde

# Joint sphérique (épaule, hanche)
engine.physics.add_ball_joint(torso, arm, anchor_a=Vec3(0.3, 0, 0))

# Joint fixe (soudure)
engine.physics.add_fixed_joint(head, torso)
```

### Zones d'eau (Flottabilité)

```python
from engine import BuoyancyZone, AABB, Vec3

water = BuoyancyZone(
    aabb=AABB(Vec3(-20, -10, -20), Vec3(20, 0, 20)),
    fluid_density=1000.0,  # eau
    fluid_drag=3.0,
)
engine.physics.add_buoyancy_zone(water)
```

### Ressorts

```python
from engine import Spring

obj_a = engine.get_object("point_a")
obj_b = engine.get_object("point_b")
spring = Spring(obj_a, obj_b, rest_length=2.0, stiffness=50.0, damping=5.0)
engine.physics.add_spring(spring)
```

## API de simulation (Step / Reset)

```python
engine = Engine()
engine.add_object("floor", Primitives.plane(), static=True)
engine.add_object("agent", Primitives.sphere(radius=0.5),
    position=Vec3(0, 5, 0), color=(0, 1, 0))

running = True
while running:
    agent = engine.get_object("agent")
    agent.rigidbody.add_force(Vec3(1, 0, 0))  # pousser l'agent
    running = engine.step(dt=1.0/60.0)

engine.reset()
```

## Lancer les tests

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=engine --cov-report=term-missing
```

## Structure du projet

```
├── main.py
├── requirements.txt
├── engine/
│   ├── math3d.py        # Vec3, Mat4
│   ├── camera.py        # Caméra FPS
│   ├── mesh.py          # Mesh + OBJLoader
│   ├── renderer.py      # Pipeline OpenGL
│   ├── engine.py        # Boucle principale, step/reset
│   ├── transform.py     # Position, rotation, échelle
│   ├── primitives.py    # Cube, sphère, cylindre, plan
│   ├── collision.py     # AABB, Ray, raycasting
│   ├── scene.py         # SceneObject
│   └── physics/
│       ├── material.py  # PhysicsMaterial + presets
│       ├── rigidbody.py # Mass, vélocité, forces
│       ├── forces.py    # Gravity, Drag, BuoyancyZone, Spring
│       ├── solver.py    # Contact, collision response
│       ├── joint.py     # HingeJoint, BallJoint, FixedJoint
│       └── world.py     # PhysicsWorld orchestrateur
├── tests/
│   ├── test_math3d.py
│   ├── test_camera.py
│   ├── test_mesh.py
│   ├── test_engine.py
│   ├── test_renderer.py
│   ├── test_main.py
│   ├── test_transform.py
│   ├── test_primitives.py
│   ├── test_collision.py
│   ├── test_scene.py
│   ├── test_material.py
│   ├── test_rigidbody.py
│   ├── test_forces.py
│   ├── test_solver.py
│   ├── test_joint.py
│   └── test_world.py
└── model/
    └── FinalBaseMesh.obj
```

## Optimisations

- Rendu GPU via OpenGL 3.3 core (vertex/fragment shaders GLSL)
- VSync, back-face culling, depth test gérés par le GPU
- Géométrie uploadée une fois en VRAM (VAO/VBO/EBO statiques)
- Cache des matrices (recalcul uniquement si modifiées)
- Physique à timestep fixe (déterministe pour RL)
- Résolution de collisions par impulsions itératives (8 itérations)
- `__slots__` sur toutes les classes
- PyOpenGL-accelerate pour des appels GL rapides