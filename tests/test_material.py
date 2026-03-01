from engine.physics.material import PhysicsMaterial
import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestPhysicsMaterialInit(unittest.TestCase):
    """Tests d'initialisation des matériaux."""

    def test_default(self):
        """Matériau par défaut."""
        m = PhysicsMaterial()
        self.assertAlmostEqual(m.friction, 0.5)
        self.assertAlmostEqual(m.restitution, 0.3)
        self.assertAlmostEqual(m.density, 1000.0)

    def test_custom(self):
        """Matériau personnalisé."""
        m = PhysicsMaterial(friction=0.8, restitution=0.9, density=500.0)
        self.assertAlmostEqual(m.friction, 0.8)
        self.assertAlmostEqual(m.restitution, 0.9)

    def test_clamped_values(self):
        """Les valeurs sont clampées aux limites."""
        m = PhysicsMaterial(friction=2.0, restitution=-0.5, density=-10.0)
        self.assertAlmostEqual(m.friction, 1.0)
        self.assertAlmostEqual(m.restitution, 0.0)
        self.assertGreater(m.density, 0.0)


class TestPhysicsMaterialPresets(unittest.TestCase):
    """Tests des presets de matériaux."""

    def test_stone(self):
        """Preset STONE a une friction élevée et faible rebond."""
        s = PhysicsMaterial.STONE
        self.assertGreater(s.friction, 0.4)
        self.assertLess(s.restitution, 0.3)

    def test_rubber(self):
        """Preset RUBBER a un rebond élevé."""
        r = PhysicsMaterial.RUBBER
        self.assertGreater(r.restitution, 0.5)

    def test_ice(self):
        """Preset ICE a très peu de friction."""
        i = PhysicsMaterial.ICE
        self.assertLess(i.friction, 0.1)

    def test_metal(self):
        """Preset METAL a une densité élevée."""
        m = PhysicsMaterial.METAL
        self.assertGreater(m.density, 5000.0)

    def test_wood(self):
        """Preset WOOD flotte dans l'eau (densité < 1000)."""
        w = PhysicsMaterial.WOOD
        self.assertLess(w.density, 1000.0)

    def test_default_preset(self):
        """Preset DEFAULT existe."""
        d = PhysicsMaterial.DEFAULT
        self.assertEqual(d.name, "default")


class TestPhysicsMaterialCombine(unittest.TestCase):
    """Tests de combinaison de matériaux."""

    def test_combine_friction(self):
        """Friction combinée est la moyenne géométrique."""
        a = PhysicsMaterial(friction=0.4)
        b = PhysicsMaterial(friction=0.9)
        combined = PhysicsMaterial.combine_friction(a, b)
        expected = math.sqrt(0.4 * 0.9)
        self.assertAlmostEqual(combined, expected, places=4)

    def test_combine_restitution(self):
        """Restitution combinée est le maximum."""
        a = PhysicsMaterial(restitution=0.3)
        b = PhysicsMaterial(restitution=0.8)
        combined = PhysicsMaterial.combine_restitution(a, b)
        self.assertAlmostEqual(combined, 0.8)

    def test_combine_friction_zero(self):
        """Friction 0 × X = 0."""
        a = PhysicsMaterial(friction=0.0)
        b = PhysicsMaterial(friction=0.9)
        self.assertAlmostEqual(PhysicsMaterial.combine_friction(a, b), 0.0)


class TestPhysicsMaterialCopy(unittest.TestCase):
    """Tests de copie de matériaux."""

    def test_copy(self):
        """La copie a les mêmes valeurs."""
        m = PhysicsMaterial(friction=0.7, restitution=0.5, name="test")
        c = m.copy()
        self.assertAlmostEqual(c.friction, 0.7)
        self.assertAlmostEqual(c.restitution, 0.5)
        self.assertEqual(c.name, "test")

    def test_copy_independent(self):
        """La copie est indépendante de l'original."""
        m = PhysicsMaterial(friction=0.5)
        c = m.copy()
        c.friction = 0.9
        self.assertAlmostEqual(m.friction, 0.5)


class TestPhysicsMaterialRepr(unittest.TestCase):
    """Tests de représentation."""

    def test_repr(self):
        """repr contient le nom."""
        m = PhysicsMaterial(name="test_mat")
        self.assertIn("test_mat", repr(m))


if __name__ == '__main__':
    unittest.main()
