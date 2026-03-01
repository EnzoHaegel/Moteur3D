"""Démo physique : robot bipède articulé avec joints moteurs.

Lance ce script pour voir un robot à deux jambes.
Les moteurs oscillent automatiquement pour simuler une marche basique.
Remplace la logique d'oscillation par un agent RL pour apprendre à marcher.
"""

import math
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))
from engine import Engine, Primitives, Vec3, PhysicsMaterial


def build_robot(engine):
    """Construit un robot bipède et retourne ses joints."""
    body_mat = PhysicsMaterial(friction=0.6, restitution=0.1, density=1200.0)
    leg_mat = PhysicsMaterial(friction=0.8, restitution=0.05, density=1000.0)

    torso = engine.add_object("torso", Primitives.cube(size=0.6),
                              position=Vec3(0, 3.5, 0), mass=5.0,
                              material=body_mat, color=(0.2, 0.7, 0.3))

    head = engine.add_object("head", Primitives.sphere(radius=0.2, segments=12, rings=12),
                             position=Vec3(0, 4.1, 0), mass=0.5,
                             material=body_mat, color=(0.9, 0.8, 0.7))
    engine.physics.add_fixed_joint(torso, head,
                                   anchor_a=Vec3(0, 0.3, 0), anchor_b=Vec3(0, -0.2, 0))

    joints = {}
    for side, x_offset in [("l", -0.2), ("r", 0.2)]:
        thigh = engine.add_object(f"thigh_{side}",
                                  Primitives.cylinder(
                                      radius=0.08, height=0.5, segments=8),
                                  position=Vec3(x_offset, 2.7, 0), mass=1.5,
                                  material=leg_mat, color=(0.15, 0.5, 0.2))

        shin = engine.add_object(f"shin_{side}",
                                 Primitives.cylinder(
                                     radius=0.06, height=0.45, segments=8),
                                 position=Vec3(x_offset, 2.0, 0), mass=1.0,
                                 material=leg_mat, color=(0.1, 0.4, 0.15))

        foot = engine.add_object(f"foot_{side}",
                                 Primitives.cube(size=0.15),
                                 position=Vec3(x_offset, 1.6, 0.05), mass=0.3,
                                 material=PhysicsMaterial.RUBBER, color=(0.3, 0.3, 0.3))

        hip = engine.physics.add_hinge_joint(torso, thigh,
                                             anchor_a=Vec3(x_offset, -0.3, 0),
                                             anchor_b=Vec3(0, 0.25, 0),
                                             axis=Vec3(1, 0, 0),
                                             min_angle=-60, max_angle=60,
                                             motor_max_force=150.0)

        knee = engine.physics.add_hinge_joint(thigh, shin,
                                              anchor_a=Vec3(0, -0.25, 0),
                                              anchor_b=Vec3(0, 0.22, 0),
                                              axis=Vec3(1, 0, 0),
                                              min_angle=-90, max_angle=0,
                                              motor_max_force=100.0)

        ankle = engine.physics.add_hinge_joint(shin, foot,
                                               anchor_a=Vec3(0, -0.22, 0),
                                               anchor_b=Vec3(0, 0.07, -0.05),
                                               axis=Vec3(1, 0, 0),
                                               min_angle=-30, max_angle=30,
                                               motor_max_force=50.0)

        hip.motor_enabled = True
        knee.motor_enabled = True
        ankle.motor_enabled = True
        joints[f"hip_{side}"] = hip
        joints[f"knee_{side}"] = knee
        joints[f"ankle_{side}"] = ankle

    return joints


def main():
    engine = Engine(title="Bipedal Robot — Joints Moteurs")

    engine.add_object("ground", Primitives.plane(width=50, depth=50),
                      static=True, material=PhysicsMaterial.STONE, color=(0.3, 0.3, 0.35))

    joints = build_robot(engine)

    step_count = 0
    freq = 2.0

    while engine.step(dt=1.0 / 60.0):
        step_count += 1
        t = step_count / 60.0

        # --- Remplace cette section par ton agent RL ---
        phase = math.sin(2.0 * math.pi * freq * t)

        joints["hip_l"].motor_speed = phase * 60.0
        joints["hip_r"].motor_speed = -phase * 60.0

        joints["knee_l"].motor_speed = (phase - 0.5) * 40.0
        joints["knee_r"].motor_speed = (-phase - 0.5) * 40.0

        joints["ankle_l"].motor_speed = phase * 20.0
        joints["ankle_r"].motor_speed = -phase * 20.0
        # ------------------------------------------------

        torso = engine.get_object("torso")
        if torso.transform.position.y < 0.3:
            print(f"Robot tombé après {t:.1f}s — reset !")
            engine.reset()
            engine.add_object("ground", Primitives.plane(width=50, depth=50),
                              static=True, material=PhysicsMaterial.STONE, color=(0.3, 0.3, 0.35))
            joints = build_robot(engine)
            step_count = 0


if __name__ == "__main__":
    main()
