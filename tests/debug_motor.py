"""Script de debug pour le moteur du HingeJoint."""
import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from engine.math3d import Vec3
from engine.physics.rigidbody import RigidBody
from engine.physics.joint import HingeJoint
from engine.scene import SceneObject
from engine.transform import Transform
from engine.mesh import Mesh


def _make_mesh():
    verts = np.array([[-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5]], dtype=np.float32)
    faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    return Mesh(verts, faces)


def _make_obj(pos, mass=1.0):
    return SceneObject(mesh=_make_mesh(), transform=Transform(position=pos), rigidbody=RigidBody(mass=mass))


parent = _make_obj(Vec3(0, 0, 0), mass=0.0)
child = _make_obj(Vec3(0, -1, 0), mass=1.0)

print(f"child rb id: {id(child.rigidbody)}")
print(f"child inv_mass: {child.rigidbody.inv_mass}")
print(f"child inv_inertia: {child.rigidbody.inv_inertia}")
print(f"child is_static: {child.rigidbody.is_static}")
print()

joint = HingeJoint(parent, child, axis=Vec3(1.0, 0.0, 0.0))
print(f"joint.obj_b is child: {joint.obj_b is child}")
print(f"joint.obj_b.rigidbody id: {id(joint.obj_b.rigidbody)}")
print(f"joint.motor_max_force: {joint.motor_max_force}")
print(f"joint.axis: {joint.axis}")
print()

joint.motor_enabled = True
joint.motor_speed = 90.0
print(f"motor_enabled: {joint.motor_enabled}")
print(f"motor_speed: {joint.motor_speed}")
print()

print("--- Direct add_angular_impulse test ---")
test_rb = RigidBody(mass=1.0)
test_rb.add_angular_impulse(Vec3(1.0, 0, 0))
print(f"Direct impulse result: angular_velocity.x = {test_rb.angular_velocity.x}")
print()

print("--- Calling _solve_motor directly ---")
child.rigidbody.angular_velocity = Vec3(0, 0, 0)
child.rigidbody.velocity = Vec3(0, 0, 0)
joint._solve_motor(1.0 / 60.0)
print(f"After _solve_motor: angular_velocity = {child.rigidbody.angular_velocity}")
print()

print("--- Calling full solve ---")
child.rigidbody.angular_velocity = Vec3(0, 0, 0)
child.rigidbody.velocity = Vec3(0, 0, 0)
joint.solve(1.0 / 60.0)
print(f"After solve: angular_velocity = {child.rigidbody.angular_velocity}")
print(f"After solve: velocity = {child.rigidbody.velocity}")
