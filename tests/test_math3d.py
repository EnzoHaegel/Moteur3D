from engine.math3d import Vec3, Mat4
import unittest
import numpy as np
import math
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestVec3(unittest.TestCase):
    """Tests unitaires pour la classe Vec3."""

    def test_creation_default(self):
        """Vec3 par défaut doit être (0, 0, 0)."""
        v = Vec3()
        self.assertAlmostEqual(v.x, 0.0)
        self.assertAlmostEqual(v.y, 0.0)
        self.assertAlmostEqual(v.z, 0.0)

    def test_creation_values(self):
        """Vec3 avec valeurs spécifiques."""
        v = Vec3(1.0, 2.0, 3.0)
        self.assertAlmostEqual(v.x, 1.0)
        self.assertAlmostEqual(v.y, 2.0)
        self.assertAlmostEqual(v.z, 3.0)

    def test_addition(self):
        """Addition de deux Vec3."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        c = a + b
        self.assertAlmostEqual(c.x, 5.0)
        self.assertAlmostEqual(c.y, 7.0)
        self.assertAlmostEqual(c.z, 9.0)

    def test_subtraction(self):
        """Soustraction de deux Vec3."""
        a = Vec3(5.0, 6.0, 7.0)
        b = Vec3(1.0, 2.0, 3.0)
        c = a - b
        self.assertAlmostEqual(c.x, 4.0)
        self.assertAlmostEqual(c.y, 4.0)
        self.assertAlmostEqual(c.z, 4.0)

    def test_scalar_multiply(self):
        """Multiplication par un scalaire."""
        v = Vec3(1.0, 2.0, 3.0)
        r = v * 2.0
        self.assertAlmostEqual(r.x, 2.0)
        self.assertAlmostEqual(r.y, 4.0)
        self.assertAlmostEqual(r.z, 6.0)

    def test_rmul(self):
        """Multiplication scalaire à gauche."""
        v = Vec3(1.0, 2.0, 3.0)
        r = 3.0 * v
        self.assertAlmostEqual(r.x, 3.0)
        self.assertAlmostEqual(r.y, 6.0)
        self.assertAlmostEqual(r.z, 9.0)

    def test_negation(self):
        """Négation d'un Vec3."""
        v = Vec3(1.0, -2.0, 3.0)
        n = -v
        self.assertAlmostEqual(n.x, -1.0)
        self.assertAlmostEqual(n.y, 2.0)
        self.assertAlmostEqual(n.z, -3.0)

    def test_dot_product(self):
        """Produit scalaire."""
        a = Vec3(1.0, 0.0, 0.0)
        b = Vec3(0.0, 1.0, 0.0)
        self.assertAlmostEqual(a.dot(b), 0.0)

        c = Vec3(1.0, 2.0, 3.0)
        d = Vec3(4.0, 5.0, 6.0)
        self.assertAlmostEqual(c.dot(d), 32.0)

    def test_cross_product(self):
        """Produit vectoriel."""
        x = Vec3(1.0, 0.0, 0.0)
        y = Vec3(0.0, 1.0, 0.0)
        z = x.cross(y)
        self.assertAlmostEqual(z.x, 0.0)
        self.assertAlmostEqual(z.y, 0.0)
        self.assertAlmostEqual(z.z, 1.0)

    def test_length(self):
        """Longueur d'un vecteur."""
        v = Vec3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v.length(), 5.0, places=5)

    def test_length_squared(self):
        """Longueur au carré d'un vecteur."""
        v = Vec3(3.0, 4.0, 0.0)
        self.assertAlmostEqual(v.length_squared(), 25.0, places=5)

    def test_normalized(self):
        """Normalisation d'un vecteur."""
        v = Vec3(3.0, 0.0, 0.0)
        n = v.normalized()
        self.assertAlmostEqual(n.x, 1.0, places=5)
        self.assertAlmostEqual(n.length(), 1.0, places=5)

    def test_normalized_zero(self):
        """Normalisation d'un vecteur nul retourne vecteur nul."""
        v = Vec3(0.0, 0.0, 0.0)
        n = v.normalized()
        self.assertAlmostEqual(n.length(), 0.0)

    def test_equality(self):
        """Egalité entre Vec3."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(1.0, 2.0, 3.0)
        self.assertEqual(a, b)

    def test_inequality(self):
        """Inégalité entre Vec3."""
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(1.0, 2.0, 4.0)
        self.assertNotEqual(a, b)

    def test_to_vec4(self):
        """Conversion en coordonnées homogènes."""
        v = Vec3(1.0, 2.0, 3.0)
        v4 = v.to_vec4(1.0)
        self.assertEqual(len(v4), 4)
        self.assertAlmostEqual(v4[3], 1.0)

    def test_from_array(self):
        """Création depuis un tableau NumPy."""
        arr = np.array([5.0, 6.0, 7.0])
        v = Vec3.from_array(arr)
        self.assertAlmostEqual(v.x, 5.0)
        self.assertAlmostEqual(v.y, 6.0)
        self.assertAlmostEqual(v.z, 7.0)

    def test_setters(self):
        """Test des setters x, y, z."""
        v = Vec3()
        v.x = 10.0
        v.y = 20.0
        v.z = 30.0
        self.assertAlmostEqual(v.x, 10.0)
        self.assertAlmostEqual(v.y, 20.0)
        self.assertAlmostEqual(v.z, 30.0)

    def test_repr(self):
        """Représentation textuelle."""
        v = Vec3(1.0, 2.0, 3.0)
        self.assertIn("Vec3", repr(v))


class TestMat4(unittest.TestCase):
    """Tests unitaires pour la classe Mat4."""

    def test_identity(self):
        """Matrice identité."""
        m = Mat4.identity()
        np.testing.assert_array_almost_equal(m.data, np.eye(4))

    def test_translation(self):
        """Matrice de translation transforme un point correctement."""
        t = Mat4.translation(5.0, 10.0, 15.0)
        p = Vec3(0.0, 0.0, 0.0)
        result = t.transform_point(p)
        self.assertAlmostEqual(result.x, 5.0, places=4)
        self.assertAlmostEqual(result.y, 10.0, places=4)
        self.assertAlmostEqual(result.z, 15.0, places=4)

    def test_scale(self):
        """Matrice de scale transforme un point correctement."""
        s = Mat4.scale(2.0, 3.0, 4.0)
        p = Vec3(1.0, 1.0, 1.0)
        result = s.transform_point(p)
        self.assertAlmostEqual(result.x, 2.0, places=4)
        self.assertAlmostEqual(result.y, 3.0, places=4)
        self.assertAlmostEqual(result.z, 4.0, places=4)

    def test_rotation_x(self):
        """Rotation 90° autour de X envoie Y sur Z."""
        r = Mat4.rotation_x(math.pi / 2)
        p = Vec3(0.0, 1.0, 0.0)
        result = r.transform_point(p)
        self.assertAlmostEqual(result.x, 0.0, places=4)
        self.assertAlmostEqual(result.y, 0.0, places=4)
        self.assertAlmostEqual(result.z, 1.0, places=4)

    def test_rotation_y(self):
        """Rotation 90° autour de Y envoie Z sur X."""
        r = Mat4.rotation_y(math.pi / 2)
        p = Vec3(0.0, 0.0, 1.0)
        result = r.transform_point(p)
        self.assertAlmostEqual(result.x, 1.0, places=4)
        self.assertAlmostEqual(result.y, 0.0, places=4)
        self.assertAlmostEqual(result.z, 0.0, places=4)

    def test_rotation_z(self):
        """Rotation 90° autour de Z envoie X sur Y."""
        r = Mat4.rotation_z(math.pi / 2)
        p = Vec3(1.0, 0.0, 0.0)
        result = r.transform_point(p)
        self.assertAlmostEqual(result.x, 0.0, places=4)
        self.assertAlmostEqual(result.y, 1.0, places=4)
        self.assertAlmostEqual(result.z, 0.0, places=4)

    def test_matmul_identity(self):
        """Multiplication par identité ne change rien."""
        a = Mat4.translation(1.0, 2.0, 3.0)
        b = Mat4.identity()
        c = a @ b
        np.testing.assert_array_almost_equal(c.data, a.data)

    def test_perspective(self):
        """La matrice perspective a les bonnes propriétés."""
        p = Mat4.perspective(math.radians(90), 1.0, 0.1, 100.0)
        self.assertAlmostEqual(p.data[3, 2], -1.0, places=4)
        self.assertAlmostEqual(p.data[3, 3], 0.0, places=4)

    def test_look_at(self):
        """look_at depuis Z=5 vers l'origine."""
        view = Mat4.look_at(
            Vec3(0.0, 0.0, 5.0),
            Vec3(0.0, 0.0, 0.0),
            Vec3(0.0, 1.0, 0.0),
        )
        p = Vec3(0.0, 0.0, 0.0)
        result = view.transform_point(p)
        self.assertAlmostEqual(abs(result.z), 5.0, places=3)

    def test_transform_points_batch(self):
        """Transformation batch de points."""
        t = Mat4.translation(1.0, 0.0, 0.0)
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        result = t.transform_points_batch(points)
        self.assertAlmostEqual(result[0, 0], 1.0, places=4)
        self.assertAlmostEqual(result[1, 0], 2.0, places=4)

    def test_combined_transform(self):
        """Chaînage translation + scale."""
        s = Mat4.scale(2.0, 2.0, 2.0)
        t = Mat4.translation(1.0, 0.0, 0.0)
        combined = t @ s
        p = Vec3(1.0, 0.0, 0.0)
        result = combined.transform_point(p)
        self.assertAlmostEqual(result.x, 3.0, places=4)


if __name__ == '__main__':
    unittest.main()
