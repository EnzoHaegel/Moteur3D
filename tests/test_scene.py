from engine.scene import SceneObject
from engine.mesh import Mesh
from engine.transform import Transform
from engine.collision import AABB
from engine.math3d import Vec3
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def _make_triangle_mesh():
    """Crée un maillage triangulaire simple."""
    verts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return Mesh(verts, faces, name="triangle")


class TestSceneObjectInit(unittest.TestCase):
    """Tests d'initialisation du SceneObject."""

    def test_default_init(self):
        """Initialisation par défaut."""
        mesh = _make_triangle_mesh()
        obj = SceneObject(mesh=mesh)
        self.assertEqual(obj.name, "object")
        self.assertTrue(obj.active)
        self.assertIsNotNone(obj.transform)

    def test_custom_init(self):
        """Initialisation avec paramètres personnalisés."""
        mesh = _make_triangle_mesh()
        t = Transform(position=Vec3(5.0, 0.0, 0.0))
        obj = SceneObject(
            mesh=mesh,
            transform=t,
            color=(1.0, 0.0, 0.0),
            name="red_triangle",
            active=False,
        )
        self.assertEqual(obj.name, "red_triangle")
        self.assertFalse(obj.active)
        self.assertEqual(obj.color, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(obj.transform.position.x, 5.0)


class TestSceneObjectAABB(unittest.TestCase):
    """Tests pour le calcul de l'AABB."""

    def test_aabb_default_transform(self):
        """AABB avec transform par défaut."""
        mesh = _make_triangle_mesh()
        obj = SceneObject(mesh=mesh)
        aabb = obj.get_aabb()
        self.assertIsInstance(aabb, AABB)
        self.assertAlmostEqual(aabb.min_point.x, 0.0, places=3)
        self.assertAlmostEqual(aabb.max_point.x, 1.0, places=3)

    def test_aabb_with_translation(self):
        """AABB avec translation appliquée."""
        mesh = _make_triangle_mesh()
        t = Transform(position=Vec3(10.0, 0.0, 0.0))
        obj = SceneObject(mesh=mesh, transform=t)
        aabb = obj.get_aabb()
        self.assertAlmostEqual(aabb.min_point.x, 10.0, places=3)
        self.assertAlmostEqual(aabb.max_point.x, 11.0, places=3)

    def test_aabb_with_scale(self):
        """AABB avec échelle appliquée."""
        mesh = _make_triangle_mesh()
        t = Transform(scale=Vec3(3.0, 3.0, 3.0))
        obj = SceneObject(mesh=mesh, transform=t)
        aabb = obj.get_aabb()
        self.assertAlmostEqual(aabb.max_point.x, 3.0, places=3)


class TestSceneObjectRepr(unittest.TestCase):
    """Tests de la représentation textuelle."""

    def test_repr(self):
        """repr() contient le nom et 'SceneObject'."""
        mesh = _make_triangle_mesh()
        obj = SceneObject(mesh=mesh, name="test_obj")
        r = repr(obj)
        self.assertIn("SceneObject", r)
        self.assertIn("test_obj", r)


if __name__ == '__main__':
    unittest.main()
