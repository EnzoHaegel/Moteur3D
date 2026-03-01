from engine.physics.solver import Contact, detect_contact, resolve_collision
from engine.physics.rigidbody import RigidBody
from engine.physics.material import PhysicsMaterial
from engine.scene import SceneObject
from engine.transform import Transform
from engine.mesh import Mesh
from engine.math3d import Vec3
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def _make_cube_mesh(size=1.0):
    """Crée un mesh cube simple."""
    h = size / 2.0
    verts = np.array([
        [-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
        [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h],
    ], dtype=np.float32)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 3, 7], [0, 7, 4],
    ], dtype=np.int32)
    return Mesh(verts, faces)


def _make_obj(pos, mass=1.0, mat=None):
    """Crée un SceneObject cube avec rigidbody."""
    mesh = _make_cube_mesh()
    t = Transform(position=pos)
    rb = RigidBody(mass=mass, material=mat)
    return SceneObject(mesh=mesh, transform=t, rigidbody=rb)


class TestDetectContact(unittest.TestCase):
    """Tests pour detect_contact."""

    def test_overlapping(self):
        """Deux cubes qui se chevauchent produisent un contact."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.8, 0.0, 0.0))
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        self.assertGreater(contact.penetration, 0.0)

    def test_separated(self):
        """Deux cubes séparés ne produisent pas de contact."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(5.0, 0.0, 0.0))
        contact = detect_contact(a, b)
        self.assertIsNone(contact)

    def test_normal_direction(self):
        """La normale pointe de A vers B."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.5, 0.0, 0.0))
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        self.assertAlmostEqual(contact.normal.x, 1.0, places=3)

    def test_vertical_collision(self):
        """La normale pointe vers le haut pour collision verticale."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.0, 0.5, 0.0))
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        self.assertAlmostEqual(abs(contact.normal.y), 1.0, places=3)


class TestResolveCollision(unittest.TestCase):
    """Tests pour resolve_collision."""

    def test_separates_objects(self):
        """La résolution sépare les vélocités."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.8, 0.0, 0.0))
        a.rigidbody.velocity = Vec3(5.0, 0.0, 0.0)
        b.rigidbody.velocity = Vec3(-5.0, 0.0, 0.0)
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        resolve_collision(contact)
        self.assertLess(a.rigidbody.velocity.x, 5.0)
        self.assertGreater(b.rigidbody.velocity.x, -5.0)

    def test_static_floor(self):
        """Un objet dynamique rebondit sur un sol statique."""
        floor = _make_obj(Vec3(0.0, -0.5, 0.0), mass=0.0)
        ball = _make_obj(Vec3(0.0, 0.4, 0.0), mass=1.0,
                         mat=PhysicsMaterial(restitution=0.5))
        ball.rigidbody.velocity = Vec3(0.0, -10.0, 0.0)
        contact = detect_contact(floor, ball)
        if contact is not None:
            resolve_collision(contact)
        self.assertGreater(ball.rigidbody.velocity.y, -10.0)

    def test_both_static_no_effect(self):
        """Deux corps statiques ne sont pas affectés."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0), mass=0.0)
        b = _make_obj(Vec3(0.5, 0.0, 0.0), mass=0.0)
        contact = detect_contact(a, b)
        if contact is not None:
            resolve_collision(contact)

    def test_rubber_bounces_more(self):
        """Le caoutchouc rebondit plus que la pierre."""
        rubber_mat = PhysicsMaterial(restitution=0.9)
        stone_mat = PhysicsMaterial(restitution=0.1)

        floor = _make_obj(Vec3(0.0, -0.5, 0.0), mass=0.0)

        rubber = _make_obj(Vec3(0.0, 0.4, 0.0), mass=1.0, mat=rubber_mat)
        rubber.rigidbody.velocity = Vec3(0.0, -10.0, 0.0)

        stone = _make_obj(Vec3(0.0, 0.4, 0.0), mass=1.0, mat=stone_mat)
        stone.rigidbody.velocity = Vec3(0.0, -10.0, 0.0)

        c_rubber = detect_contact(floor, rubber)
        c_stone = detect_contact(floor, stone)
        if c_rubber:
            resolve_collision(c_rubber)
        if c_stone:
            resolve_collision(c_stone)

        self.assertGreater(rubber.rigidbody.velocity.y,
                           stone.rigidbody.velocity.y)

    def test_no_rigidbody_safe(self):
        """Pas de crash si un objet n'a pas de rigidbody."""
        mesh = _make_cube_mesh()
        a = SceneObject(mesh=mesh, transform=Transform(position=Vec3(0, 0, 0)))
        b = _make_obj(Vec3(0.5, 0.0, 0.0))
        contact = detect_contact(a, b)
        if contact is not None:
            resolve_collision(contact)


class TestContactInfo(unittest.TestCase):
    """Tests pour le Contact."""

    def test_contact_fields(self):
        """Les champs du contact sont correctement remplis."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.8, 0.0, 0.0))
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        self.assertIs(contact.obj_a, a)
        self.assertIs(contact.obj_b, b)
        self.assertIsNotNone(contact.point)
        self.assertGreater(contact.penetration, 0.0)


if __name__ == '__main__':
    unittest.main()
