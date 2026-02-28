from engine.camera import Camera
from engine.math3d import Vec3, Mat4
import unittest
import math
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestCamera(unittest.TestCase):
    """Tests unitaires pour la classe Camera."""

    def setUp(self):
        """Initialise une caméra pour chaque test."""
        self.cam = Camera(
            position=Vec3(0.0, 0.0, 0.0),
            yaw=-90.0,
            pitch=0.0,
        )

    def test_initial_position(self):
        """Position initiale correcte."""
        self.assertAlmostEqual(self.cam.position.x, 0.0)
        self.assertAlmostEqual(self.cam.position.y, 0.0)
        self.assertAlmostEqual(self.cam.position.z, 0.0)

    def test_initial_forward(self):
        """Direction avant initiale (yaw=-90) pointe vers Z négatif."""
        self.assertAlmostEqual(self.cam.forward.x, 0.0, places=4)
        self.assertAlmostEqual(self.cam.forward.z, -1.0, places=4)

    def test_mouse_yaw(self):
        """Le mouvement horizontal de la souris modifie le yaw."""
        initial_yaw = self.cam.yaw
        self.cam.process_mouse(100.0, 0.0)
        self.assertGreater(self.cam.yaw, initial_yaw)

    def test_mouse_pitch_clamped(self):
        """Le pitch est clampé entre -89 et 89 degrés."""
        self.cam.process_mouse(0.0, -10000.0)
        self.assertLessEqual(self.cam.pitch, 89.0)

        self.cam.process_mouse(0.0, 10000.0)
        self.assertGreaterEqual(self.cam.pitch, -89.0)

    def test_move_forward(self):
        """Déplacement avant avec la touche Z."""
        initial_pos = Vec3(self.cam.position.x,
                           self.cam.position.y, self.cam.position.z)
        keys = {'z': True, 's': False, 'q': False,
                'd': False, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys, 1.0)
        self.assertNotEqual(self.cam.position, initial_pos)

    def test_move_backward(self):
        """Déplacement arrière avec la touche S."""
        keys_fwd = {'z': True, 's': False, 'q': False,
                    'd': False, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys_fwd, 1.0)
        pos_fwd = self.cam.position

        self.cam.position = Vec3(0.0, 0.0, 0.0)
        keys_bwd = {'z': False, 's': True, 'q': False,
                    'd': False, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys_bwd, 1.0)
        pos_bwd = self.cam.position

        self.assertAlmostEqual(pos_fwd.x + pos_bwd.x, 0.0, places=3)
        self.assertAlmostEqual(pos_fwd.z + pos_bwd.z, 0.0, places=3)

    def test_move_left_right(self):
        """Déplacement latéral Q et D."""
        keys_left = {'z': False, 's': False, 'q': True,
                     'd': False, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys_left, 1.0)
        pos_left = Vec3(self.cam.position.x,
                        self.cam.position.y, self.cam.position.z)

        self.cam.position = Vec3(0.0, 0.0, 0.0)
        keys_right = {'z': False, 's': False, 'q': False,
                      'd': True, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys_right, 1.0)
        pos_right = self.cam.position

        self.assertAlmostEqual(pos_left.x + pos_right.x, 0.0, places=3)

    def test_move_up_down(self):
        """Déplacement vertical avec Space et Shift."""
        keys_up = {'z': False, 's': False, 'q': False,
                   'd': False, 'space': True, 'shift': False}
        self.cam.process_keyboard(keys_up, 1.0)
        self.assertGreater(self.cam.position.y, 0.0)

    def test_no_movement_no_keys(self):
        """Aucun déplacement si aucune touche pressée."""
        keys = {'z': False, 's': False, 'q': False,
                'd': False, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys, 1.0)
        self.assertAlmostEqual(self.cam.position.x, 0.0)
        self.assertAlmostEqual(self.cam.position.y, 0.0)
        self.assertAlmostEqual(self.cam.position.z, 0.0)

    def test_view_matrix_type(self):
        """La matrice de vue est bien un Mat4."""
        view = self.cam.get_view_matrix()
        self.assertIsInstance(view, Mat4)

    def test_projection_matrix_type(self):
        """La matrice de projection est bien un Mat4."""
        proj = self.cam.get_projection_matrix()
        self.assertIsInstance(proj, Mat4)

    def test_vp_matrix_type(self):
        """La matrice VP combinée est bien un Mat4."""
        vp = self.cam.get_vp_matrix()
        self.assertIsInstance(vp, Mat4)

    def test_view_matrix_cached(self):
        """La matrice de vue est mise en cache."""
        v1 = self.cam.get_view_matrix()
        v2 = self.cam.get_view_matrix()
        self.assertIs(v1, v2)

    def test_view_matrix_invalidated_on_move(self):
        """La matrice de vue est recalculée après déplacement."""
        v1 = self.cam.get_view_matrix()
        keys = {'z': True, 's': False, 'q': False,
                'd': False, 'space': False, 'shift': False}
        self.cam.process_keyboard(keys, 0.1)
        v2 = self.cam.get_view_matrix()
        self.assertIsNot(v1, v2)

    def test_speed_property(self):
        """La vitesse est modifiable."""
        self.cam.speed = 20.0
        self.assertAlmostEqual(self.cam.speed, 20.0)

    def test_fov_property(self):
        """Le champ de vision est modifiable."""
        self.cam.fov = 90.0
        self.assertAlmostEqual(self.cam.fov, 90.0)

    def test_aspect_property(self):
        """Le ratio d'aspect est modifiable."""
        self.cam.aspect = 2.0
        self.assertAlmostEqual(self.cam.aspect, 2.0)


if __name__ == '__main__':
    unittest.main()
