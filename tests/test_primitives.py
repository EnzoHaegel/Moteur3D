from engine.primitives import Primitives
import unittest
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestCube(unittest.TestCase):
    """Tests pour la primitive cube."""

    def test_vertex_count(self):
        """Un cube a 24 sommets (4 par face × 6 faces)."""
        mesh = Primitives.cube()
        self.assertEqual(mesh.vertex_count(), 24)

    def test_face_count(self):
        """Un cube a 12 faces triangulaires (2 par face × 6 faces)."""
        mesh = Primitives.cube()
        self.assertEqual(mesh.face_count(), 12)

    def test_normals_computed(self):
        """Les normales sont calculées."""
        mesh = Primitives.cube()
        self.assertIsNotNone(mesh.normals)
        self.assertEqual(mesh.normals.shape[0], 24)

    def test_name(self):
        """Le nom du mesh est 'cube'."""
        mesh = Primitives.cube()
        self.assertEqual(mesh.name, "cube")

    def test_custom_size(self):
        """La taille personnalisée est respectée."""
        mesh = Primitives.cube(size=4.0)
        bmin, bmax = mesh.get_bounds()
        self.assertAlmostEqual(bmax[0] - bmin[0], 4.0, places=4)

    def test_centered(self):
        """Le cube est centré à l'origine."""
        mesh = Primitives.cube()
        center = mesh.get_center()
        self.assertAlmostEqual(center[0], 0.0, places=4)
        self.assertAlmostEqual(center[1], 0.0, places=4)
        self.assertAlmostEqual(center[2], 0.0, places=4)


class TestSphere(unittest.TestCase):
    """Tests pour la primitive sphère."""

    def test_vertex_count(self):
        """Nombre de sommets correct pour segments=8, rings=8."""
        mesh = Primitives.sphere(segments=8, rings=8)
        expected = (8 + 1) * (8 + 1)
        self.assertEqual(mesh.vertex_count(), expected)

    def test_face_count(self):
        """Nombre de faces correct pour segments=8, rings=8."""
        mesh = Primitives.sphere(segments=8, rings=8)
        expected = 8 * 8 * 2
        self.assertEqual(mesh.face_count(), expected)

    def test_normals_computed(self):
        """Les normales sont calculées."""
        mesh = Primitives.sphere()
        self.assertIsNotNone(mesh.normals)

    def test_name(self):
        """Le nom du mesh est 'sphere'."""
        mesh = Primitives.sphere()
        self.assertEqual(mesh.name, "sphere")

    def test_radius_bounds(self):
        """Les vertices sont à la bonne distance du centre."""
        r = 2.5
        mesh = Primitives.sphere(radius=r, segments=16, rings=16)
        distances = np.linalg.norm(mesh.vertices, axis=1)
        np.testing.assert_array_almost_equal(distances, r, decimal=4)

    def test_default_radius(self):
        """La sphère par défaut a un rayon de 1."""
        mesh = Primitives.sphere()
        bmin, bmax = mesh.get_bounds()
        self.assertAlmostEqual(bmax[1], 1.0, places=4)


class TestCylinder(unittest.TestCase):
    """Tests pour la primitive cylindre."""

    def test_has_vertices(self):
        """Le cylindre a des sommets."""
        mesh = Primitives.cylinder()
        self.assertGreater(mesh.vertex_count(), 0)

    def test_has_faces(self):
        """Le cylindre a des faces."""
        mesh = Primitives.cylinder()
        self.assertGreater(mesh.face_count(), 0)

    def test_normals_computed(self):
        """Les normales sont calculées."""
        mesh = Primitives.cylinder()
        self.assertIsNotNone(mesh.normals)

    def test_name(self):
        """Le nom du mesh est 'cylinder'."""
        mesh = Primitives.cylinder()
        self.assertEqual(mesh.name, "cylinder")

    def test_height(self):
        """La hauteur correspond au paramètre."""
        h = 5.0
        mesh = Primitives.cylinder(height=h)
        bmin, bmax = mesh.get_bounds()
        actual_h = bmax[1] - bmin[1]
        self.assertAlmostEqual(actual_h, h, places=4)

    def test_segments(self):
        """Le nombre de segments produit le bon nombre de faces latérales."""
        seg = 8
        mesh = Primitives.cylinder(segments=seg)
        lateral = seg * 2
        caps = seg * 2
        self.assertEqual(mesh.face_count(), lateral + caps)


class TestPlane(unittest.TestCase):
    """Tests pour la primitive plan."""

    def test_vertex_count(self):
        """Un plan a 4 sommets."""
        mesh = Primitives.plane()
        self.assertEqual(mesh.vertex_count(), 4)

    def test_face_count(self):
        """Un plan a 2 faces triangulaires."""
        mesh = Primitives.plane()
        self.assertEqual(mesh.face_count(), 2)

    def test_normals_point_up(self):
        """Les normales du plan pointent vers Y positif."""
        mesh = Primitives.plane()
        for n in mesh.normals:
            self.assertAlmostEqual(abs(n[1]), 1.0, places=4)

    def test_name(self):
        """Le nom du mesh est 'plane'."""
        mesh = Primitives.plane()
        self.assertEqual(mesh.name, "plane")

    def test_dimensions(self):
        """Les dimensions correspondent aux paramètres."""
        mesh = Primitives.plane(width=6.0, depth=4.0)
        bmin, bmax = mesh.get_bounds()
        self.assertAlmostEqual(bmax[0] - bmin[0], 6.0, places=4)
        self.assertAlmostEqual(bmax[2] - bmin[2], 4.0, places=4)

    def test_y_is_zero(self):
        """Tous les sommets sont à Y=0."""
        mesh = Primitives.plane()
        for v in mesh.vertices:
            self.assertAlmostEqual(v[1], 0.0, places=6)


if __name__ == '__main__':
    unittest.main()
