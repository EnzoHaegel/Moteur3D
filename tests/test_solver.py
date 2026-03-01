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

    def test_horizontal_x_collision(self):
        """La normale pointe sur l'axe X pour collision latérale X."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.5, 0.0, 0.0))
        # Ensure overlap on X is strictly smaller than Y and Z
        # a: X (-0.5 to 0.5), b: X (0.0 to 1.0), Overlap X: 0.5
        # a: Y (-0.5 to 0.5), b: Y (-0.5 to 0.5), Overlap Y: 1.0
        # a: Z (-0.5 to 0.5), b: Z (-0.5 to 0.5), Overlap Z: 1.0
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        self.assertAlmostEqual(abs(contact.normal.x), 1.0, places=3)
        self.assertEqual(contact.normal.y, 0.0)
        self.assertEqual(contact.normal.z, 0.0)
        self.assertAlmostEqual(contact.penetration, 0.5, places=3)

    def test_flat_plane_collision(self):
        """Un objet traversant un plan plat (sans épaisseur) génère un contact valide (épaisseur 0)."""
        from engine.primitives import Primitives
        plane_mesh = Primitives.plane(width=10, depth=10)
        plane_obj = SceneObject(mesh=plane_mesh, transform=Transform(position=Vec3(0, 0, 0)))
        
        # Objet descendant à moitié dans le plan
        cube_obj = _make_obj(Vec3(0, 0.0, 0.0)) # Cube fait 1.0 de haut, de -0.5 à +0.5
        cube_obj.rigidbody.velocity = Vec3(0, -5.0, 0)
        
        contact = detect_contact(plane_obj, cube_obj)
        self.assertIsNotNone(contact)
        self.assertGreater(contact.penetration, 0.0)
        self.assertAlmostEqual(contact.penetration, 0.5, places=3)
        self.assertAlmostEqual(contact.normal.y, -1.0, places=3) # Normale face vers le bas car on descend ? Non, plan statique
        
    def test_thin_separation(self):
        """Tests that penetration calculation uses minimum translation distance, not overlap size."""
        # Sol mince (H=0.0) de y=0.0 à y=0.0
        plane_mesh = Mesh(np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32), 
                          np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32))
        plane_obj = SceneObject(mesh=plane_mesh, transform=Transform(position=Vec3(0, 0, 0)))
        
        # Cube (H=1.0) encastré de y=-0.2 à y=0.8
        cube = _make_obj(Vec3(0, 0.3, 0.0))
        
        contact = detect_contact(plane_obj, cube)
        self.assertIsNotNone(contact)
        # Size of intersection on Y is 0.0 (plane max 0.0 - plane min 0.0) 
        # But penetration should be 0.2 (dist to separate)
        self.assertAlmostEqual(contact.penetration, 0.2, places=3)


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
            
    def test_zero_inv_mass_sum(self):
        """Deux objets avec inv_mass=0 renvoient early (inv_mass_sum==0)."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0), mass=1.0)
        b = _make_obj(Vec3(0.5, 0.0, 0.0), mass=1.0)
        
        # Pour forcer inv_mass_sum == 0.0 sans déclencher is_static && is_static
        # (car is_static = mass == 0.0)
        a.rigidbody._inv_mass = 0.0
        b.rigidbody._inv_mass = 0.0
        
        contact = detect_contact(a, b)
        if contact is not None:
            resolve_collision(contact)
            
    def test_separating_velocity(self):
        """Objets qui s'éloignent ne provoquent qu'une correction de position, pas d'impulsion."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.5, 0.0, 0.0))
        # B moves away from A
        a.rigidbody.velocity = Vec3(-10.0, 0.0, 0.0)
        b.rigidbody.velocity = Vec3(10.0, 0.0, 0.0)
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        resolve_collision(contact)
        # Velocity should not change if they separate
        self.assertEqual(a.rigidbody.velocity.x, -10.0)
        self.assertEqual(b.rigidbody.velocity.x, 10.0)

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

    def test_horizontal_z_collision(self):
        """La normale pointe sur l'axe Z pour collision horizontale en profondeur."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0))
        b = _make_obj(Vec3(0.0, 0.0, 0.5))
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        self.assertAlmostEqual(abs(contact.normal.z), 1.0, places=3)
        self.assertEqual(contact.normal.x, 0.0)
        self.assertEqual(contact.normal.y, 0.0)
        
    def test_friction_tangent_impulse(self):
        """Vérifie que la friction ralentit la vitesse tangentielle (glissement)."""
        # Un sol avec beaucoup de friction
        floor_mat = PhysicsMaterial(friction=1.0)
        floor = _make_obj(Vec3(0.0, -0.5, 0.0), mass=0.0, mat=floor_mat)
        
        # Un bloc qui glisse avec friction
        block_mat = PhysicsMaterial(friction=1.0)
        block = _make_obj(Vec3(0.0, 0.49, 0.0), mass=1.0, mat=block_mat)
        
        # Le bloc tombe légèrement (pour la normale) ET avance sur le côté X (tangentiel)
        block.rigidbody.velocity = Vec3(10.0, -2.0, 0.0)
        
        contact = detect_contact(floor, block)
        self.assertIsNotNone(contact)
        
        resolve_collision(contact)
        
        # Vitesse tangentielle X devrait avoir diminuée à cause de la friction
        self.assertLess(block.rigidbody.velocity.x, 10.0)
        # La gravité/rebond aura affecté Y
        self.assertGreater(block.rigidbody.velocity.y, -2.0)

    def test_no_rigidbody_safe(self):
        """Pas de crash si un objet n'a pas de rigidbody."""
        mesh = _make_cube_mesh()
        a = SceneObject(mesh=mesh, transform=Transform(position=Vec3(0, 0, 0)))
        b = _make_obj(Vec3(0.5, 0.0, 0.0))
        contact = detect_contact(a, b)
        if contact is not None:
            resolve_collision(contact)
            
        a2 = _make_obj(Vec3(0.0, 0.0, 0.0))
        b2 = SceneObject(mesh=mesh, transform=Transform(position=Vec3(0.5, 0, 0)))
        contact2 = detect_contact(a2, b2)
        if contact2 is not None:
            resolve_collision(contact2)

    def test_zero_tangent_friction(self):
        """Aucune erreur si la vélocité tangentielle est très petite (chute verticale pure)."""
        floor = _make_obj(Vec3(0.0, -0.5, 0.0), mass=0.0)
        block = _make_obj(Vec3(0.0, 0.49, 0.0), mass=1.0)
        block.rigidbody.velocity = Vec3(0.0, -10.0, 0.0)  # Perfectly straight down
        
        contact = detect_contact(floor, block)
        self.assertIsNotNone(contact)
        resolve_collision(contact)
        # Should not crash, lateral velocity should remain 0
        self.assertEqual(block.rigidbody.velocity.x, 0.0)
        self.assertEqual(block.rigidbody.velocity.z, 0.0)
        
    def test_clamped_friction(self):
        """Vérifie le block 'else' où la friction est clampée par -normal_impulse * mu."""
        # Un sol normal
        floor_mat = PhysicsMaterial(friction=0.1) # low friction
        floor = _make_obj(Vec3(0.0, -0.5, 0.0), mass=0.0, mat=floor_mat)
        
        # Un bloc très rapide (fort jt mais clampé par mu bas)
        block_mat = PhysicsMaterial(friction=0.1)
        block = _make_obj(Vec3(0.0, 0.49, 0.0), mass=1.0, mat=block_mat)
        block.rigidbody.velocity = Vec3(50.0, -1.0, 0.0) # Vitesse extreme latérale
        
        contact = detect_contact(floor, block)
        self.assertIsNotNone(contact)
        resolve_collision(contact)
        # Should have applied reduced friction
        self.assertLess(block.rigidbody.velocity.x, 50.0)
        self.assertGreater(block.rigidbody.velocity.x, 0.0)  # Still mostly 50 since friction is low

    def test_correct_position_static_a(self):
        """Vérifie que la position de A n'est pas modifiée si A est statique (B dynamique)."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0), mass=0.0)  # A est statique
        b = _make_obj(Vec3(0.5, 0.0, 0.0), mass=1.0)  # B est dynamique
        b.rigidbody.velocity = Vec3(-10.0, 0.0, 0.0)
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        resolve_collision(contact)
        self.assertEqual(a.transform.position.x, 0.0)  # A ne bouge pas
        self.assertGreater(b.transform.position.x, 0.5)  # B est repoussé

    def test_correct_position_static_b(self):
        """Vérifie que la position de B n'est pas modifiée si B est statique (A dynamique)."""
        a = _make_obj(Vec3(0.0, 0.0, 0.0), mass=1.0)  # A est dynamique
        a.rigidbody.velocity = Vec3(10.0, 0.0, 0.0)
        b = _make_obj(Vec3(0.5, 0.0, 0.0), mass=0.0)  # B est statique
        contact = detect_contact(a, b)
        self.assertIsNotNone(contact)
        resolve_collision(contact)
        self.assertLess(a.transform.position.x, 0.0)  # A est repoussé
        self.assertEqual(b.transform.position.x, 0.5)  # B ne bouge pas


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
