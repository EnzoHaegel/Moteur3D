"""Démo physique : objets dans l'eau avec flottabilité.

Lance ce script pour voir des objets flotter ou couler
selon leur densité et le matériau appliqué.
Le bois flotte, le métal coule, le caoutchouc flotte à mi-hauteur.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from engine import Engine, Primitives, Vec3, PhysicsMaterial, BuoyancyZone, AABB


def main():
    engine = Engine(title="Swimming — Flottabilité")

    engine.add_object("seabed", Primitives.plane(width=40, depth=40),
                      position=Vec3(0, -8, 0), static=True,
                      material=PhysicsMaterial.STONE, color=(0.4, 0.35, 0.25))

    water_surface = engine.add_object("water_surface",
                                      Primitives.plane(width=40, depth=40),
                                      position=Vec3(0, 0, 0), static=True,
                                      color=(0.1, 0.3, 0.8))
    engine.physics.unregister(water_surface)
    water_surface.rigidbody = None

    water_zone = BuoyancyZone(
        aabb=AABB(Vec3(-20, -8, -20), Vec3(20, 0, 20)),
        fluid_density=1000.0,
        fluid_drag=3.0,
    )
    engine.physics.add_buoyancy_zone(water_zone)

    engine.add_object("wood_log", Primitives.cylinder(radius=0.3, height=2.0, segments=8),
                      position=Vec3(-4, 3, 0), mass=0.6 * 0.3 * 0.3 * 3.14 * 2.0,
                      material=PhysicsMaterial.WOOD, color=(0.6, 0.4, 0.2))

    engine.add_object("metal_block", Primitives.cube(size=0.8),
                      position=Vec3(0, 4, 0), mass=7800.0 * 0.8**3,
                      material=PhysicsMaterial.METAL, color=(0.7, 0.7, 0.75))

    engine.add_object("rubber_ball", Primitives.sphere(radius=0.4, segments=12, rings=12),
                      position=Vec3(4, 5, 0), mass=1100.0 * (4/3) * 3.14 * 0.4**3,
                      material=PhysicsMaterial.RUBBER, color=(1.0, 0.3, 0.3))

    engine.add_object("ice_cube", Primitives.cube(size=0.6),
                      position=Vec3(-2, 6, 2), mass=917.0 * 0.6**3,
                      material=PhysicsMaterial.ICE, color=(0.7, 0.95, 1.0))

    engine.add_object("beach_ball", Primitives.sphere(radius=0.5, segments=12, rings=12),
                      position=Vec3(2, 2, -2), mass=0.1,
                      material=PhysicsMaterial(friction=0.3, restitution=0.6, density=50.0,
                                               name="plastic"),
                      color=(1.0, 1.0, 0.2))

    engine.run()


if __name__ == "__main__":
    main()
