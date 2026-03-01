from engine.collision import AABB, Ray, ray_aabb_intersect
from engine.math3d import Vec3
from engine.mesh import Mesh
from engine.transform import Transform
import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestAABBInit(unittest.TestCase):
    """Tests d'initialisation de l'AABB."""

    def test_min_max(self):
        """Min et max sont stockés correctement."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        self.assertEqual(aabb.min_point, Vec3(0.0, 0.0, 0.0))
        self.assertEqual(aabb.max_point, Vec3(1.0, 1.0, 1.0))

    def test_center(self):
        """Le centre est calculé correctement."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 4.0, 6.0))
        c = aabb.center()
        self.assertAlmostEqual(c.x, 1.0)
        self.assertAlmostEqual(c.y, 2.0)
        self.assertAlmostEqual(c.z, 3.0)

    def test_size(self):
        """Les dimensions sont calculées correctement."""
        aabb = AABB(Vec3(1.0, 2.0, 3.0), Vec3(4.0, 6.0, 9.0))
        s = aabb.size()
        self.assertAlmostEqual(s.x, 3.0)
        self.assertAlmostEqual(s.y, 4.0)
        self.assertAlmostEqual(s.z, 6.0)

    def test_repr(self):
        """repr() contient 'AABB'."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        self.assertIn("AABB", repr(aabb))


class TestAABBIntersects(unittest.TestCase):
    """Tests d'intersection entre AABB."""

    def test_overlapping(self):
        """Deux AABB qui se chevauchent."""
        a = AABB(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0))
        b = AABB(Vec3(1.0, 1.0, 1.0), Vec3(3.0, 3.0, 3.0))
        self.assertTrue(a.intersects(b))
        self.assertTrue(b.intersects(a))

    def test_separated(self):
        """Deux AABB séparées."""
        a = AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        b = AABB(Vec3(5.0, 5.0, 5.0), Vec3(6.0, 6.0, 6.0))
        self.assertFalse(a.intersects(b))

    def test_touching(self):
        """Deux AABB qui se touchent sur une face."""
        a = AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        b = AABB(Vec3(1.0, 0.0, 0.0), Vec3(2.0, 1.0, 1.0))
        self.assertTrue(a.intersects(b))

    def test_contained(self):
        """Une AABB contenue dans une autre."""
        a = AABB(Vec3(0.0, 0.0, 0.0), Vec3(10.0, 10.0, 10.0))
        b = AABB(Vec3(2.0, 2.0, 2.0), Vec3(3.0, 3.0, 3.0))
        self.assertTrue(a.intersects(b))

    def test_separated_on_one_axis(self):
        """Séparées sur un seul axe."""
        a = AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        b = AABB(Vec3(0.0, 0.0, 2.0), Vec3(1.0, 1.0, 3.0))
        self.assertFalse(a.intersects(b))


class TestAABBContainsPoint(unittest.TestCase):
    """Tests de contenance d'un point."""

    def test_inside(self):
        """Point à l'intérieur de l'AABB."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0))
        self.assertTrue(aabb.contains_point(Vec3(1.0, 1.0, 1.0)))

    def test_outside(self):
        """Point à l'extérieur de l'AABB."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0))
        self.assertFalse(aabb.contains_point(Vec3(5.0, 5.0, 5.0)))

    def test_on_boundary(self):
        """Point sur le bord de l'AABB."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(2.0, 2.0, 2.0))
        self.assertTrue(aabb.contains_point(Vec3(2.0, 1.0, 1.0)))

    def test_at_min_corner(self):
        """Point au coin minimum."""
        aabb = AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))
        self.assertTrue(aabb.contains_point(Vec3(0.0, 0.0, 0.0)))


class TestAABBFromMesh(unittest.TestCase):
    """Tests pour AABB.from_mesh."""

    def _make_triangle_mesh(self):
        """Crée un triangle simple."""
        verts = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return Mesh(verts, faces)

    def test_from_mesh_no_transform(self):
        """AABB d'un mesh sans transformation."""
        mesh = self._make_triangle_mesh()
        aabb = AABB.from_mesh(mesh)
        self.assertAlmostEqual(aabb.min_point.x, 0.0)
        self.assertAlmostEqual(aabb.max_point.x, 2.0)
        self.assertAlmostEqual(aabb.max_point.y, 3.0)

    def test_from_mesh_with_translation(self):
        """AABB d'un mesh avec translation."""
        mesh = self._make_triangle_mesh()
        t = Transform(position=Vec3(10.0, 0.0, 0.0))
        aabb = AABB.from_mesh(mesh, t)
        self.assertAlmostEqual(aabb.min_point.x, 10.0, places=3)
        self.assertAlmostEqual(aabb.max_point.x, 12.0, places=3)

    def test_from_mesh_with_scale(self):
        """AABB d'un mesh avec échelle."""
        mesh = self._make_triangle_mesh()
        t = Transform(scale=Vec3(2.0, 2.0, 2.0))
        aabb = AABB.from_mesh(mesh, t)
        self.assertAlmostEqual(aabb.max_point.x, 4.0, places=3)
        self.assertAlmostEqual(aabb.max_point.y, 6.0, places=3)


class TestRay(unittest.TestCase):
    """Tests pour la classe Ray."""

    def test_direction_normalized(self):
        """La direction est normalisée."""
        r = Ray(Vec3(0.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0))
        self.assertAlmostEqual(r.direction.length(), 1.0, places=5)

    def test_point_at(self):
        """point_at retourne le bon point."""
        r = Ray(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
        p = r.point_at(5.0)
        self.assertAlmostEqual(p.x, 5.0, places=4)
        self.assertAlmostEqual(p.y, 0.0, places=4)

    def test_repr(self):
        """repr() contient 'Ray'."""
        r = Ray(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0))
        self.assertIn("Ray", repr(r))


class TestRayAABBIntersect(unittest.TestCase):
    """Tests pour ray_aabb_intersect."""

    def _unit_aabb(self):
        """AABB unité de (0,0,0) à (1,1,1)."""
        return AABB(Vec3(0.0, 0.0, 0.0), Vec3(1.0, 1.0, 1.0))

    def test_hit_from_front(self):
        """Rayon qui frappe l'AABB de face."""
        ray = Ray(Vec3(-5.0, 0.5, 0.5), Vec3(1.0, 0.0, 0.0))
        t = ray_aabb_intersect(ray, self._unit_aabb())
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 5.0, places=4)

    def test_miss(self):
        """Rayon qui rate l'AABB."""
        ray = Ray(Vec3(-5.0, 5.0, 5.0), Vec3(1.0, 0.0, 0.0))
        t = ray_aabb_intersect(ray, self._unit_aabb())
        self.assertIsNone(t)

    def test_ray_behind(self):
        """Rayon qui pointe dans la direction opposée."""
        ray = Ray(Vec3(-5.0, 0.5, 0.5), Vec3(-1.0, 0.0, 0.0))
        t = ray_aabb_intersect(ray, self._unit_aabb())
        self.assertIsNone(t)

    def test_ray_inside(self):
        """Rayon dont l'origine est à l'intérieur de l'AABB."""
        ray = Ray(Vec3(0.5, 0.5, 0.5), Vec3(1.0, 0.0, 0.0))
        t = ray_aabb_intersect(ray, self._unit_aabb())
        self.assertIsNotNone(t)
        self.assertGreaterEqual(t, 0.0)

    def test_parallel_miss(self):
        """Rayon parallèle à une face mais à l'extérieur."""
        ray = Ray(Vec3(-1.0, 5.0, 0.5), Vec3(1.0, 0.0, 0.0))
        t = ray_aabb_intersect(ray, self._unit_aabb())
        self.assertIsNone(t)

    def test_diagonal_hit(self):
        """Rayon en diagonale qui frappe l'AABB."""
        ray = Ray(Vec3(-1.0, -1.0, -1.0), Vec3(1.0, 1.0, 1.0))
        t = ray_aabb_intersect(ray, self._unit_aabb())
        self.assertIsNotNone(t)


if __name__ == '__main__':
    unittest.main()
