from engine.physics.forces import Gravity, Drag, BuoyancyZone, Spring
from engine.physics.rigidbody import RigidBody
from engine.physics.material import PhysicsMaterial
from engine.collision import AABB
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


def _make_mesh():
    """Crée un mesh simple pour les tests."""
    verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
    ], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return Mesh(verts, faces)


def _make_scene_object(pos=None, mass=1.0):
    """Crée un SceneObject avec rigidbody pour les tests."""
    mesh = _make_mesh()
    t = Transform(position=pos if pos else Vec3(0.0, 0.0, 0.0))
    rb = RigidBody(mass=mass)
    return SceneObject(mesh=mesh, transform=t, rigidbody=rb)


class TestGravity(unittest.TestCase):
    """Tests pour la gravité."""

    def test_default_direction(self):
        """Gravité par défaut vers le bas."""
        g = Gravity()
        self.assertAlmostEqual(g.acceleration.y, -9.81, places=4)

    def test_apply(self):
        """La gravité ajoute une force vers le bas."""
        g = Gravity()
        rb = RigidBody(mass=2.0)
        g.apply(rb)
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.y, -9.81, places=2)

    def test_gravity_scale(self):
        """gravity_scale multiplie la force."""
        g = Gravity()
        rb = RigidBody(mass=1.0)
        rb.gravity_scale = 0.5
        g.apply(rb)
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.y, -9.81 * 0.5, places=2)

    def test_static_ignored(self):
        """Les corps statiques ne sont pas affectés."""
        g = Gravity()
        rb = RigidBody(mass=0.0)
        g.apply(rb)
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.y, 0.0)

    def test_custom_gravity(self):
        """Gravité personnalisée (planète)."""
        g = Gravity(Vec3(0.0, -3.7, 0.0))
        rb = RigidBody(mass=1.0)
        g.apply(rb)
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.y, -3.7, places=2)


class TestDrag(unittest.TestCase):
    """Tests pour la résistance de l'air."""

    def test_no_drag_at_rest(self):
        """Pas de traînée à vélocité nulle."""
        d = Drag()
        rb = RigidBody()
        d.apply(rb)
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.x, 0.0)

    def test_drag_slows_down(self):
        """La traînée réduit la vélocité."""
        d = Drag()
        rb = RigidBody()
        rb.material.drag = 0.1
        rb.velocity = Vec3(10.0, 0.0, 0.0)
        d.apply(rb)
        rb.integrate_forces(0.1)
        self.assertLess(rb.velocity.x, 10.0)

    def test_static_ignored(self):
        """Les corps statiques ne sont pas affectés."""
        d = Drag()
        rb = RigidBody(mass=0.0)
        d.apply(rb)


class TestBuoyancyZone(unittest.TestCase):
    """Tests pour la flottabilité."""

    def test_submerged_gets_upward_force(self):
        """Un objet submergé reçoit une force vers le haut."""
        zone = BuoyancyZone(
            aabb=AABB(Vec3(-10, -10, -10), Vec3(10, 0, 10)),
            fluid_density=1000.0,
        )
        rb = RigidBody(mass=1.0)
        obj_aabb = AABB(Vec3(-0.5, -2.0, -0.5), Vec3(0.5, -1.0, 0.5))
        zone.apply(rb, obj_aabb)
        rb.integrate_forces(1.0)
        self.assertGreater(rb.velocity.y, 0.0)

    def test_above_water_no_force(self):
        """Un objet au-dessus de l'eau ne reçoit pas de force."""
        zone = BuoyancyZone(
            aabb=AABB(Vec3(-10, -10, -10), Vec3(10, 0, 10)),
        )
        rb = RigidBody(mass=1.0)
        obj_aabb = AABB(Vec3(-0.5, 1.0, -0.5), Vec3(0.5, 2.0, 0.5))
        zone.apply(rb, obj_aabb)
        rb.integrate_forces(1.0)
        self.assertAlmostEqual(rb.velocity.y, 0.0)

    def test_partial_submersion(self):
        """Un objet partiellement submergé reçoit une force intermédiaire."""
        zone = BuoyancyZone(
            aabb=AABB(Vec3(-10, -10, -10), Vec3(10, 0, 10)),
            fluid_density=1000.0,
        )
        rb_full = RigidBody(mass=1.0)
        rb_half = RigidBody(mass=1.0)
        full_aabb = AABB(Vec3(-0.5, -2.0, -0.5), Vec3(0.5, -1.0, 0.5))
        half_aabb = AABB(Vec3(-0.5, -0.5, -0.5), Vec3(0.5, 0.5, 0.5))
        zone.apply(rb_full, full_aabb)
        zone.apply(rb_half, half_aabb)
        rb_full.integrate_forces(1.0)
        rb_half.integrate_forces(1.0)
        self.assertGreater(rb_full.velocity.y, rb_half.velocity.y)


class TestSpring(unittest.TestCase):
    """Tests pour les ressorts."""

    def test_spring_pulls_together(self):
        """Le ressort rapproche les objets si étirés."""
        obj_a = _make_scene_object(Vec3(0.0, 0.0, 0.0))
        obj_b = _make_scene_object(Vec3(5.0, 0.0, 0.0))
        spring = Spring(obj_a, obj_b, rest_length=1.0,
                        stiffness=100.0, damping=0.0)
        spring.apply()
        obj_a.rigidbody.integrate_forces(0.1)
        obj_b.rigidbody.integrate_forces(0.1)
        self.assertGreater(obj_a.rigidbody.velocity.x, 0.0)
        self.assertLess(obj_b.rigidbody.velocity.x, 0.0)

    def test_spring_auto_rest_length(self):
        """rest_length auto-calculée depuis les positions."""
        obj_a = _make_scene_object(Vec3(0.0, 0.0, 0.0))
        obj_b = _make_scene_object(Vec3(3.0, 0.0, 0.0))
        spring = Spring(obj_a, obj_b)
        self.assertAlmostEqual(spring.rest_length, 3.0, places=3)

    def test_spring_at_rest_no_force(self):
        """Pas de force quand le ressort est à sa longueur de repos."""
        obj_a = _make_scene_object(Vec3(0.0, 0.0, 0.0))
        obj_b = _make_scene_object(Vec3(2.0, 0.0, 0.0))
        spring = Spring(obj_a, obj_b, rest_length=2.0,
                        stiffness=100.0, damping=0.0)
        spring.apply()
        obj_a.rigidbody.integrate_forces(0.1)
        self.assertAlmostEqual(obj_a.rigidbody.velocity.x, 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
