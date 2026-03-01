from engine.physics.joint import HingeJoint, BallJoint, FixedJoint, _rotate_vec_by_euler
from engine.physics.rigidbody import RigidBody
from engine.scene import SceneObject
from engine.transform import Transform
from engine.mesh import Mesh
from engine.math3d import Vec3
import unittest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def _make_mesh():
    """Crée un mesh simple."""
    verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return Mesh(verts, faces)


def _make_obj(pos, mass=1.0):
    """Crée un SceneObject avec rigidbody."""
    mesh = _make_mesh()
    t = Transform(position=pos)
    rb = RigidBody(mass=mass)
    return SceneObject(mesh=mesh, transform=t, rigidbody=rb)


class TestRotateVecByEuler(unittest.TestCase):
    """Tests pour la rotation d'un vecteur par angles d'Euler."""

    def test_identity_rotation(self):
        """Rotation nulle retourne le même vecteur."""
        v = Vec3(1.0, 0.0, 0.0)
        result = _rotate_vec_by_euler(v, Vec3(0.0, 0.0, 0.0))
        self.assertAlmostEqual(result.x, 1.0, places=5)
        self.assertAlmostEqual(result.y, 0.0, places=5)
        self.assertAlmostEqual(result.z, 0.0, places=5)

    def test_90_deg_y_rotation(self):
        """Rotation de 90° autour de Y envoie X vers -Z."""
        v = Vec3(1.0, 0.0, 0.0)
        result = _rotate_vec_by_euler(v, Vec3(0.0, 90.0, 0.0))
        self.assertAlmostEqual(result.x, 0.0, places=5)
        self.assertAlmostEqual(result.y, 0.0, places=5)
        self.assertAlmostEqual(result.z, -1.0, places=5)


class TestHingeJoint(unittest.TestCase):
    """Tests pour le HingeJoint."""

    def test_position_constraint(self):
        """Le joint maintient les points d'ancrage proches."""
        parent = _make_obj(Vec3(0.0, 2.0, 0.0), mass=0.0)
        child = _make_obj(Vec3(0.0, -2.0, 0.0))
        joint = HingeJoint(
            parent, child,
            anchor_a=Vec3(0.0, -1.0, 0.0),
            anchor_b=Vec3(0.0, 0.5, 0.0),
        )
        for _ in range(10):
            joint.solve(1.0 / 60.0)
        vel = child.rigidbody.velocity
        self.assertGreater(vel.y, 0.0)

    def test_motor(self):
        """Le moteur applique un couple."""
        parent = _make_obj(Vec3(0.0, 0.0, 0.0), mass=0.0)
        child = _make_obj(Vec3(0.0, -1.0, 0.0))
        joint = HingeJoint(parent, child, axis=Vec3(1.0, 0.0, 0.0))
        joint.motor_enabled = True
        joint.motor_speed = 90.0
        for _ in range(10):
            joint.solve(1.0 / 60.0)
        self.assertNotAlmostEqual(child.rigidbody.angular_velocity.x, 0.0)

    def test_inactive_no_effect(self):
        """Un joint inactif n'a pas d'effet."""
        parent = _make_obj(Vec3(0.0, 2.0, 0.0), mass=0.0)
        child = _make_obj(Vec3(0.0, 0.0, 0.0))
        joint = HingeJoint(parent, child, anchor_a=Vec3(0, -1, 0))
        joint.active = False
        joint.solve(1.0 / 60.0)
        self.assertAlmostEqual(child.rigidbody.velocity.y, 0.0)

    def test_repr(self):
        """repr contient 'HingeJoint'."""
        parent = _make_obj(Vec3(0, 0, 0))
        child = _make_obj(Vec3(0, -1, 0))
        joint = HingeJoint(parent, child)
        self.assertIn("HingeJoint", repr(joint))

    def test_current_angle(self):
        """current_angle est accessible."""
        parent = _make_obj(Vec3(0, 0, 0))
        child = _make_obj(Vec3(0, -1, 0))
        joint = HingeJoint(parent, child, axis=Vec3(1, 0, 0))
        joint.solve(1.0 / 60.0)
        self.assertIsInstance(joint.current_angle, float)

    def test_motor_reaction_on_parent(self):
        """Le moteur applique un couple de réaction sur le parent."""
        parent = _make_obj(Vec3(0.0, 0.0, 0.0), mass=2.0)
        child = _make_obj(Vec3(0.0, -1.0, 0.0), mass=1.0)
        joint = HingeJoint(parent, child, axis=Vec3(1.0, 0.0, 0.0))
        joint.motor_enabled = True
        joint.motor_speed = 90.0
        for _ in range(10):
            joint.solve(1.0 / 60.0)
        self.assertNotAlmostEqual(parent.rigidbody.angular_velocity.x, 0.0)

    def test_rotated_anchor(self):
        """Les ancres suivent la rotation de l'objet."""
        parent = _make_obj(Vec3(0, 0, 0), mass=0.0)
        parent.transform.rotation = Vec3(0, 90, 0)
        child = _make_obj(Vec3(0, 0, -1), mass=1.0)
        joint = HingeJoint(
            parent, child,
            anchor_a=Vec3(0, 0, -1),
            anchor_b=Vec3(0, 0, 0),
        )
        world_a = joint._get_world_anchor_a()
        self.assertAlmostEqual(world_a.x, -1.0, places=4)
        self.assertAlmostEqual(world_a.z, 0.0, places=4)


class TestBallJoint(unittest.TestCase):
    """Tests pour le BallJoint."""

    def test_position_constraint(self):
        """Le joint sphérique maintient les ancres proches."""
        parent = _make_obj(Vec3(0.0, 2.0, 0.0), mass=0.0)
        child = _make_obj(Vec3(0.0, -2.0, 0.0))
        joint = BallJoint(
            parent, child,
            anchor_a=Vec3(0.0, -1.0, 0.0),
            anchor_b=Vec3(0.0, 0.5, 0.0),
        )
        for _ in range(10):
            joint.solve(1.0 / 60.0)
        self.assertGreater(child.rigidbody.velocity.y, 0.0)

    def test_inactive(self):
        """Un joint inactif n'a pas d'effet."""
        parent = _make_obj(Vec3(0, 2, 0), mass=0.0)
        child = _make_obj(Vec3(0, 0, 0))
        joint = BallJoint(parent, child)
        joint.active = False
        joint.solve(1.0 / 60.0)
        self.assertAlmostEqual(child.rigidbody.velocity.y, 0.0)

    def test_repr(self):
        """repr contient 'BallJoint'."""
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(1, 0, 0))
        joint = BallJoint(a, b)
        self.assertIn("BallJoint", repr(joint))


class TestFixedJoint(unittest.TestCase):
    """Tests pour le FixedJoint."""

    def test_position_constraint(self):
        """Le joint fixe maintient les objets ensemble."""
        parent = _make_obj(Vec3(0.0, 2.0, 0.0), mass=0.0)
        child = _make_obj(Vec3(0.0, -2.0, 0.0))
        joint = FixedJoint(
            parent, child,
            anchor_a=Vec3(0.0, -1.0, 0.0),
            anchor_b=Vec3(0.0, 0.5, 0.0),
        )
        for _ in range(10):
            joint.solve(1.0 / 60.0)
        self.assertGreater(child.rigidbody.velocity.y, 0.0)

    def test_rotation_constraint(self):
        """Le joint fixe résiste à la rotation relative."""
        parent = _make_obj(Vec3(0, 0, 0), mass=0.0)
        child = _make_obj(Vec3(0, 0, 0))
        child.transform.rotation = Vec3(45.0, 0.0, 0.0)
        joint = FixedJoint(parent, child)
        for _ in range(10):
            joint.solve(1.0 / 60.0)
        self.assertNotAlmostEqual(child.rigidbody.angular_velocity.x, 0.0)

    def test_repr(self):
        """repr contient 'FixedJoint'."""
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(1, 0, 0))
        joint = FixedJoint(a, b)
        self.assertIn("FixedJoint", repr(joint))


if __name__ == '__main__':
    unittest.main()
