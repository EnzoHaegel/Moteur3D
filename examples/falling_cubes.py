"""Démo physique : cubes qui tombent avec différents matériaux.

Lance ce script pour voir des cubes tomber sur un sol et rebondir
différemment selon leur matériau (caoutchouc, pierre, glace, bois, métal).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from engine import Engine, Primitives, Vec3, PhysicsMaterial


def main():
    engine = Engine(title="Falling Cubes — Matériaux")

    engine.add_object("ground", Primitives.plane(width=30, depth=30),
                      static=True, material=PhysicsMaterial.STONE, color=(0.25, 0.25, 0.3))

    materials = [
        ("rubber",  PhysicsMaterial.RUBBER, (1.0, 0.2, 0.2), Vec3(-6, 8, 0)),
        ("stone",   PhysicsMaterial.STONE,  (0.6, 0.6, 0.6), Vec3(-3, 10, 0)),
        ("ice",     PhysicsMaterial.ICE,    (0.5, 0.9, 1.0), Vec3(0, 12, 0)),
        ("wood",    PhysicsMaterial.WOOD,   (0.7, 0.5, 0.3), Vec3(3, 14, 0)),
        ("metal",   PhysicsMaterial.METAL,  (0.8, 0.8, 0.85), Vec3(6, 16, 0)),
    ]

    for name, mat, color, pos in materials:
        engine.add_object(name, Primitives.cube(size=1.0),
                          position=pos, mass=mat.density * 0.001,
                          material=mat, color=color)

    engine.run()


if __name__ == "__main__":
    main()
