from engine.transform import Transform
from engine.math3d import Vec3, Mat4
import unittest
import math
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestTransformInit(unittest.TestCase):
    """Tests d'initialisation du Transform."""

    def test_default_position(self):
        """Position par défaut à l'origine."""
        t = Transform()
        self.assertEqual(t.position, Vec3(0.0, 0.0, 0.0))

    def test_default_rotation(self):
        """Rotation par défaut nulle."""
        t = Transform()
        self.assertEqual(t.rotation, Vec3(0.0, 0.0, 0.0))

    def test_default_scale(self):
        """Échelle par défaut unitaire."""
        t = Transform()
        self.assertEqual(t.scale, Vec3(1.0, 1.0, 1.0))

    def test_custom_init(self):
        """Initialisation avec valeurs personnalisées."""
        t = Transform(
            position=Vec3(1.0, 2.0, 3.0),
            rotation=Vec3(45.0, 90.0, 0.0),
            scale=Vec3(2.0, 2.0, 2.0),
        )
        self.assertEqual(t.position, Vec3(1.0, 2.0, 3.0))
        self.assertEqual(t.rotation, Vec3(45.0, 90.0, 0.0))
        self.assertEqual(t.scale, Vec3(2.0, 2.0, 2.0))


class TestTransformMatrix(unittest.TestCase):
    """Tests de la matrice modèle TRS."""

    def test_identity_at_default(self):
        """Transform par défaut produit la matrice identité."""
        t = Transform()
        m = t.get_model_matrix()
        np.testing.assert_array_almost_equal(m.data, np.eye(4), decimal=5)

    def test_translation_only(self):
        """Matrice avec translation pure."""
        t = Transform(position=Vec3(5.0, 10.0, 15.0))
        m = t.get_model_matrix()
        p = m.transform_point(Vec3(0.0, 0.0, 0.0))
        self.assertAlmostEqual(p.x, 5.0, places=4)
        self.assertAlmostEqual(p.y, 10.0, places=4)
        self.assertAlmostEqual(p.z, 15.0, places=4)

    def test_scale_only(self):
        """Matrice avec échelle pure."""
        t = Transform(scale=Vec3(2.0, 3.0, 4.0))
        m = t.get_model_matrix()
        p = m.transform_point(Vec3(1.0, 1.0, 1.0))
        self.assertAlmostEqual(p.x, 2.0, places=4)
        self.assertAlmostEqual(p.y, 3.0, places=4)
        self.assertAlmostEqual(p.z, 4.0, places=4)

    def test_rotation_90_y(self):
        """Rotation 90° autour de Y envoie Z sur X."""
        t = Transform(rotation=Vec3(0.0, 90.0, 0.0))
        m = t.get_model_matrix()
        p = m.transform_point(Vec3(0.0, 0.0, 1.0))
        self.assertAlmostEqual(p.x, 1.0, places=4)
        self.assertAlmostEqual(p.z, 0.0, places=4)

    def test_combined_trs(self):
        """Translation + Scale combinées."""
        t = Transform(
            position=Vec3(1.0, 0.0, 0.0),
            scale=Vec3(2.0, 2.0, 2.0),
        )
        m = t.get_model_matrix()
        p = m.transform_point(Vec3(1.0, 0.0, 0.0))
        self.assertAlmostEqual(p.x, 3.0, places=4)


class TestTransformCaching(unittest.TestCase):
    """Tests du cache de la matrice."""

    def test_matrix_cached(self):
        """La matrice est réutilisée si pas de changement."""
        t = Transform()
        m1 = t.get_model_matrix()
        m2 = t.get_model_matrix()
        self.assertIs(m1, m2)

    def test_matrix_invalidated_on_position_change(self):
        """La matrice est recalculée après changement de position."""
        t = Transform()
        m1 = t.get_model_matrix()
        t.position = Vec3(1.0, 0.0, 0.0)
        m2 = t.get_model_matrix()
        self.assertIsNot(m1, m2)

    def test_matrix_invalidated_on_rotation_change(self):
        """La matrice est recalculée après changement de rotation."""
        t = Transform()
        m1 = t.get_model_matrix()
        t.rotation = Vec3(0.0, 90.0, 0.0)
        m2 = t.get_model_matrix()
        self.assertIsNot(m1, m2)

    def test_matrix_invalidated_on_scale_change(self):
        """La matrice est recalculée après changement d'échelle."""
        t = Transform()
        m1 = t.get_model_matrix()
        t.scale = Vec3(2.0, 2.0, 2.0)
        m2 = t.get_model_matrix()
        self.assertIsNot(m1, m2)


class TestTransformTranslate(unittest.TestCase):
    """Tests de la méthode translate."""

    def test_translate_offset(self):
        """translate() déplace la position correctement."""
        t = Transform(position=Vec3(1.0, 2.0, 3.0))
        t.translate(Vec3(10.0, 0.0, 0.0))
        self.assertAlmostEqual(t.position.x, 11.0, places=4)
        self.assertAlmostEqual(t.position.y, 2.0, places=4)

    def test_translate_invalidates_cache(self):
        """translate() invalide le cache de matrice."""
        t = Transform()
        m1 = t.get_model_matrix()
        t.translate(Vec3(1.0, 0.0, 0.0))
        m2 = t.get_model_matrix()
        self.assertIsNot(m1, m2)


class TestTransformRepr(unittest.TestCase):
    """Tests de la représentation textuelle."""

    def test_repr(self):
        """repr() contient 'Transform'."""
        t = Transform()
        self.assertIn("Transform", repr(t))


if __name__ == '__main__':
    unittest.main()
