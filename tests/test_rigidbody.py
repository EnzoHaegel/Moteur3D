from engine.physics.rigidbody import RigidBody
from engine.physics.material import PhysicsMaterial
from engine.math3d import Vec3
import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestRigidBodyInit(unittest.TestCase):
    """Tests d'initialisation du RigidBody."""

    def test_default(self):
        """Corps par défaut avec masse 1."""
        rb = RigidBody()
        self.assertAlmostEqual(rb.mass, 1.0)
        self.assertAlmostEqual(rb.inv_mass, 1.0)
        self.assertFalse(rb.is_static)

    def test_static(self):
        """Corps statique (masse 0)."""
        rb = RigidBody(mass=0.0)
        self.assertTrue(rb.is_static)
        self.assertAlmostEqual(rb.inv_mass, 0.0)

    def test_kinematic(self):
        """Corps kinématique a une masse inverse de 0."""
        rb = RigidBody(mass=5.0, is_kinematic=True)
        self.assertAlmostEqual(rb.inv_mass, 0.0)
        self.assertTrue(rb.is_static)

    def test_custom_material(self):
        """Matériau personnalisé est stocké."""
        mat = PhysicsMaterial.RUBBER
        rb = RigidBody(material=mat)
        self.assertEqual(rb.material.name, "rubber")

    def test_default_material(self):
        """Un matériau par défaut est attribué."""
        rb = RigidBody()
        self.assertIsNotNone(rb.material)

    def test_initial_velocity_zero(self):
        """Vélocité initiale à zéro."""
        rb = RigidBody()
        self.assertEqual(rb.velocity, Vec3(0.0, 0.0, 0.0))


class TestRigidBodyForces(unittest.TestCase):
    """Tests d'application de forces."""

    def test_add_force(self):
        """add_force accumule les forces."""
        rb = RigidBody()
        rb.add_force(Vec3(10.0, 0.0, 0.0))
        rb.add_force(Vec3(0.0, 5.0, 0.0))
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.x, 10.0, places=3)
        self.assertAlmostEqual(rb.velocity.y, 5.0, places=3)

    def test_add_force_static_ignored(self):
        """Les forces sont ignorées pour les corps statiques."""
        rb = RigidBody(mass=0.0)
        rb.add_force(Vec3(100.0, 0.0, 0.0))
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.x, 0.0)

    def test_add_torque(self):
        """add_torque accumule les couples."""
        rb = RigidBody()
        rb.add_torque(Vec3(5.0, 0.0, 0.0))
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.angular_velocity.x, 5.0, places=3)

    def test_clear_forces(self):
        """clear_forces remet à zéro."""
        rb = RigidBody()
        rb.add_force(Vec3(10.0, 0.0, 0.0))
        rb.clear_forces()
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.x, 0.0)


class TestRigidBodyImpulse(unittest.TestCase):
    """Tests d'impulsions."""

    def test_add_impulse(self):
        """L'impulsion modifie directement la vélocité."""
        rb = RigidBody(mass=2.0)
        rb.add_impulse(Vec3(10.0, 0.0, 0.0))
        self.assertAlmostEqual(rb.velocity.x, 5.0, places=3)

    def test_add_impulse_static_ignored(self):
        """L'impulsion est ignorée pour les corps statiques."""
        rb = RigidBody(mass=0.0)
        rb.add_impulse(Vec3(100.0, 0.0, 0.0))
        self.assertAlmostEqual(rb.velocity.x, 0.0)

    def test_add_angular_impulse(self):
        """L'impulsion angulaire modifie la vélocité angulaire."""
        rb = RigidBody(mass=1.0)
        rb.add_angular_impulse(Vec3(0.0, 3.0, 0.0))
        self.assertAlmostEqual(rb.angular_velocity.y, 3.0, places=3)


class TestRigidBodyIntegration(unittest.TestCase):
    """Tests d'intégration."""

    def test_integrate_velocity(self):
        """La vélocité déplace la position."""
        rb = RigidBody()
        rb.velocity = Vec3(10.0, 0.0, 0.0)
        pos = Vec3(0.0, 0.0, 0.0)
        rot = Vec3(0.0, 0.0, 0.0)
        new_pos, new_rot = rb.integrate_velocity(pos, rot, 0.5)
        self.assertAlmostEqual(new_pos.x, 5.0, places=3)

    def test_integrate_static_no_move(self):
        """Les corps statiques ne bougent pas."""
        rb = RigidBody(mass=0.0)
        rb.velocity = Vec3(10.0, 0.0, 0.0)
        pos = Vec3(1.0, 2.0, 3.0)
        rot = Vec3(0.0, 0.0, 0.0)
        new_pos, new_rot = rb.integrate_velocity(pos, rot, 1.0)
        self.assertAlmostEqual(new_pos.x, 1.0)

    def test_gravity_effect(self):
        """Force de gravité modifie la vélocité en Y."""
        rb = RigidBody(mass=1.0)
        rb.add_force(Vec3(0.0, -9.81, 0.0))
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.y, -9.81, places=2)


class TestRigidBodyKineticEnergy(unittest.TestCase):
    """Tests d'énergie cinétique."""

    def test_kinetic_energy(self):
        """E = 0.5 * m * v²."""
        rb = RigidBody(mass=2.0)
        rb.velocity = Vec3(3.0, 4.0, 0.0)
        ke = rb.kinetic_energy()
        self.assertAlmostEqual(ke, 0.5 * 2.0 * 25.0, places=3)

    def test_static_zero_energy(self):
        """Les corps statiques ont 0 énergie."""
        rb = RigidBody(mass=0.0)
        self.assertAlmostEqual(rb.kinetic_energy(), 0.0)


class TestRigidBodyRepr(unittest.TestCase):
    """Tests de représentation."""

    def test_repr(self):
        """repr contient 'RigidBody'."""
        rb = RigidBody()
        self.assertIn("RigidBody", repr(rb))


if __name__ == '__main__':
    unittest.main()
