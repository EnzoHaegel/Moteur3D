from engine.physics.world import PhysicsWorld
from engine.physics.rigidbody import RigidBody
from engine.physics.material import PhysicsMaterial
from engine.physics.forces import BuoyancyZone, Spring
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


def _make_cube_mesh(size=1.0):
    """Crée un mesh cube."""
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
    """Crée un SceneObject avec rigidbody."""
    mesh = _make_cube_mesh()
    t = Transform(position=pos)
    rb = RigidBody(mass=mass, material=mat)
    return SceneObject(mesh=mesh, transform=t, rigidbody=rb)


class TestPhysicsWorldInit(unittest.TestCase):
    """Tests d'initialisation du monde physique."""

    def test_default(self):
        """Monde physique par défaut."""
        pw = PhysicsWorld()
        self.assertIsNotNone(pw.gravity)
        self.assertGreater(pw.fixed_dt, 0.0)

    def test_custom_gravity(self):
        """Gravité personnalisée."""
        pw = PhysicsWorld(gravity=Vec3(0.0, -3.7, 0.0))
        self.assertAlmostEqual(pw.gravity.acceleration.y, -3.7)

    def test_fixed_dt_setter(self):
        """Le fixed_dt est modifiable."""
        pw = PhysicsWorld()
        pw.fixed_dt = 1.0 / 60.0
        self.assertAlmostEqual(pw.fixed_dt, 1.0 / 60.0)


class TestPhysicsWorldRegistration(unittest.TestCase):
    """Tests d'enregistrement d'objets."""

    def test_register(self):
        """Enregistrement d'un objet."""
        pw = PhysicsWorld()
        obj = _make_obj(Vec3(0, 0, 0))
        pw.register(obj)
        self.assertEqual(len(pw._objects), 1)

    def test_unregister(self):
        """Désenregistrement d'un objet."""
        pw = PhysicsWorld()
        obj = _make_obj(Vec3(0, 0, 0))
        pw.register(obj)
        pw.unregister(obj)
        self.assertEqual(len(pw._objects), 0)

    def test_no_duplicate(self):
        """Pas de doublons."""
        pw = PhysicsWorld()
        obj = _make_obj(Vec3(0, 0, 0))
        pw.register(obj)
        pw.register(obj)
        self.assertEqual(len(pw._objects), 1)


class TestPhysicsWorldFalling(unittest.TestCase):
    """Tests de chute libre."""

    def test_object_falls(self):
        """Un objet tombe sous la gravité."""
        pw = PhysicsWorld()
        obj = _make_obj(Vec3(0.0, 10.0, 0.0))
        pw.register(obj)
        for _ in range(60):
            pw.step(1.0 / 60.0)
        self.assertLess(obj.transform.position.y, 10.0)

    def test_static_stays(self):
        """Un objet statique ne bouge pas."""
        pw = PhysicsWorld()
        obj = _make_obj(Vec3(0.0, 0.0, 0.0), mass=0.0)
        pw.register(obj)
        for _ in range(60):
            pw.step(1.0 / 60.0)
        self.assertAlmostEqual(obj.transform.position.y, 0.0, places=3)


class TestPhysicsWorldCollision(unittest.TestCase):
    """Tests de collision dans le monde."""

    def test_object_lands_on_floor(self):
        """Un objet atterrit sur un sol statique."""
        pw = PhysicsWorld()
        floor = _make_obj(Vec3(0.0, -1.0, 0.0), mass=0.0)
        ball = _make_obj(Vec3(0.0, 2.0, 0.0), mass=1.0)
        pw.register(floor)
        pw.register(ball)
        for _ in range(300):
            pw.step(1.0 / 60.0)
        self.assertGreater(ball.transform.position.y, -1.5)


class TestPhysicsWorldJoints(unittest.TestCase):
    """Tests de joints dans le monde."""

    def test_add_hinge_joint(self):
        """Ajout d'un joint charnière."""
        pw = PhysicsWorld()
        a = _make_obj(Vec3(0, 2, 0), mass=0.0)
        b = _make_obj(Vec3(0, 0, 0))
        pw.register(a)
        pw.register(b)
        joint = pw.add_hinge_joint(a, b, anchor_a=Vec3(0, -1, 0))
        self.assertEqual(len(pw.joints), 1)
        self.assertIs(joint.obj_a, a)

    def test_add_ball_joint(self):
        """Ajout d'un joint sphérique."""
        pw = PhysicsWorld()
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(1, 0, 0))
        joint = pw.add_ball_joint(a, b)
        self.assertEqual(len(pw.joints), 1)

    def test_add_fixed_joint(self):
        """Ajout d'un joint fixe."""
        pw = PhysicsWorld()
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(0, -1, 0))
        joint = pw.add_fixed_joint(a, b)
        self.assertEqual(len(pw.joints), 1)

    def test_remove_joint(self):
        """Suppression d'un joint."""
        pw = PhysicsWorld()
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(0, -1, 0))
        joint = pw.add_hinge_joint(a, b)
        pw.remove_joint(joint)
        self.assertEqual(len(pw.joints), 0)


class TestPhysicsWorldBuoyancy(unittest.TestCase):
    """Tests de flottabilité dans le monde."""

    def test_buoyancy_zone(self):
        """Un objet dans l'eau reçoit une force de flottabilité."""
        pw = PhysicsWorld()
        zone = BuoyancyZone(
            aabb=AABB(Vec3(-10, -10, -10), Vec3(10, 0, 10)),
            fluid_density=1000.0,
        )
        pw.add_buoyancy_zone(zone)
        self.assertEqual(len(pw.buoyancy_zones), 1)


class TestPhysicsWorldSpring(unittest.TestCase):
    """Tests de ressorts dans le monde."""

    def test_add_spring(self):
        """Ajout d'un ressort."""
        pw = PhysicsWorld()
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(3, 0, 0))
        spring = Spring(a, b, stiffness=50.0)
        pw.add_spring(spring)
        self.assertEqual(len(pw.springs), 1)


class TestPhysicsWorldReset(unittest.TestCase):
    """Tests de reset du monde physique."""

    def test_reset_clears_all(self):
        """reset() vide tout."""
        pw = PhysicsWorld()
        obj = _make_obj(Vec3(0, 0, 0))
        pw.register(obj)
        a = _make_obj(Vec3(0, 0, 0))
        b = _make_obj(Vec3(1, 0, 0))
        pw.add_hinge_joint(a, b)
        pw.add_buoyancy_zone(BuoyancyZone(
            aabb=AABB(Vec3(-1, -1, -1), Vec3(1, 1, 1))))
        pw.add_spring(Spring(a, b))
        pw.reset()
        self.assertEqual(len(pw._objects), 0)
        self.assertEqual(len(pw.joints), 0)
        self.assertEqual(len(pw.buoyancy_zones), 0)
        self.assertEqual(len(pw.springs), 0)


if __name__ == '__main__':
    unittest.main()
