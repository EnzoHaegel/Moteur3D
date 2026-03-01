import engine.engine
import pygame
import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def _make_mock_renderer():
    """Crée un mock de Renderer avec overlay surface mockée."""
    renderer = MagicMock()
    renderer.width = 800
    renderer.height = 600
    renderer.overlay = MagicMock()
    return renderer


def _create_engine_patched(**kwargs):
    """Crée un Engine en mockant pygame.display et le Renderer OpenGL."""
    with patch('engine.engine.pygame.display') as mock_display, \
            patch('engine.engine.Renderer') as MockRenderer, \
            patch('engine.engine.pygame.mouse') as mock_mouse, \
            patch('engine.engine.pygame.event') as mock_event:
        mock_display.set_mode.return_value = MagicMock()
        mock_display.gl_set_attribute = MagicMock()
        MockRenderer.return_value = _make_mock_renderer()
        mock_mouse.set_visible = MagicMock()
        mock_event.set_grab = MagicMock()

        from engine.engine import Engine
        engine = Engine(**kwargs)
    return engine


class TestEngineInit(unittest.TestCase):
    """Tests d'initialisation de l'Engine."""

    @patch('engine.engine.pygame.display')
    @patch('engine.engine.Renderer')
    @patch('engine.engine.pygame.mouse')
    @patch('engine.engine.pygame.event')
    def test_default_init(self, mock_event, mock_mouse, MockRenderer, mock_display):
        MockRenderer.return_value = _make_mock_renderer()
        from engine.engine import Engine
        engine = Engine()
        mock_display.set_caption.assert_called_with("Moteur 3D")

    @patch('engine.engine.pygame.display')
    @patch('engine.engine.Renderer')
    @patch('engine.engine.pygame.mouse')
    @patch('engine.engine.pygame.event')
    def test_custom_init(self, mock_event, mock_mouse, MockRenderer, mock_display):
        MockRenderer.return_value = _make_mock_renderer()
        from engine.engine import Engine
        engine = Engine(width=1920, height=1080, title="Test")
        mock_display.set_caption.assert_called_with("Test")


class TestEngineProperties(unittest.TestCase):
    """Tests des propriétés camera et renderer."""

    def setUp(self):
        self.engine = _create_engine_patched()

    def test_camera_type(self):
        from engine.camera import Camera
        self.assertIsInstance(self.engine.camera, Camera)

    def test_renderer_accessible(self):
        self.assertIsNotNone(self.engine.renderer)


class TestEngineMeshManagement(unittest.TestCase):
    """Tests pour load_mesh et add_mesh."""

    def setUp(self):
        self.engine = _create_engine_patched()

    def test_add_mesh_without_model(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        self.engine.add_mesh(mesh)
        self.assertEqual(len(self.engine._meshes), 1)

    def test_add_mesh_with_model(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        model = Mat4.scale(2.0, 2.0, 2.0)
        self.engine.add_mesh(mesh, model)
        self.assertEqual(len(self.engine._model_matrices), 1)

    def test_load_mesh(self):
        import tempfile
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        fd, path = tempfile.mkstemp(suffix='.obj')
        with os.fdopen(fd, 'w') as f:
            f.write(obj_content)
        try:
            mesh = self.engine.load_mesh(path)
            self.assertEqual(mesh.vertex_count(), 3)
            self.assertEqual(len(self.engine._meshes), 1)
        finally:
            os.unlink(path)

    def test_load_mesh_with_model(self):
        import tempfile
        from engine.math3d import Mat4
        obj_content = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        fd, path = tempfile.mkstemp(suffix='.obj')
        with os.fdopen(fd, 'w') as f:
            f.write(obj_content)
        try:
            model = Mat4.translation(1.0, 2.0, 3.0)
            mesh = self.engine.load_mesh(path, model)
            self.assertEqual(len(self.engine._model_matrices), 1)
        finally:
            os.unlink(path)


class TestEngineHandleEvents(unittest.TestCase):
    """Tests pour _handle_events."""

    def setUp(self):
        self.engine = _create_engine_patched()

    @patch('engine.engine.pygame.event')
    def test_quit_event(self, mock_event):
        quit_evt = MagicMock()
        quit_evt.type = pygame.QUIT
        mock_event.get.return_value = [quit_evt]
        result = self.engine._handle_events()
        self.assertFalse(result)

    @patch('engine.engine.pygame.event')
    def test_escape_releases_mouse(self, mock_event):
        esc_evt = MagicMock()
        esc_evt.type = pygame.KEYDOWN
        esc_evt.key = pygame.K_ESCAPE
        mock_event.get.return_value = [esc_evt]
        self.engine._mouse_captured = True
        result = self.engine._handle_events()
        self.assertTrue(result)
        self.assertFalse(self.engine._mouse_captured)

    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.mouse')
    def test_escape_quits_when_not_captured(self, mock_mouse, mock_event):
        esc_evt = MagicMock()
        esc_evt.type = pygame.KEYDOWN
        esc_evt.key = pygame.K_ESCAPE
        mock_event.get.return_value = [esc_evt]
        self.engine._mouse_captured = False
        result = self.engine._handle_events()
        self.assertFalse(result)

    @patch('engine.engine.pygame.event')
    def test_f1_toggles_render_mode(self, mock_event):
        f1_evt = MagicMock()
        f1_evt.type = pygame.KEYDOWN
        f1_evt.key = pygame.K_F1
        mock_event.get.return_value = [f1_evt]
        self.engine._render_mode = 'solid'
        self.engine._handle_events()
        self.assertEqual(self.engine._render_mode, 'wireframe')

    @patch('engine.engine.pygame.event')
    def test_f1_toggles_back_to_solid(self, mock_event):
        f1_evt = MagicMock()
        f1_evt.type = pygame.KEYDOWN
        f1_evt.key = pygame.K_F1
        mock_event.get.return_value = [f1_evt]
        self.engine._render_mode = 'wireframe'
        self.engine._handle_events()
        self.assertEqual(self.engine._render_mode, 'solid')

    @patch('engine.engine.pygame.event')
    def test_f2_toggles_grid(self, mock_event):
        f2_evt = MagicMock()
        f2_evt.type = pygame.KEYDOWN
        f2_evt.key = pygame.K_F2
        mock_event.get.return_value = [f2_evt]
        self.engine._show_grid = True
        self.engine._handle_events()
        self.assertFalse(self.engine._show_grid)

    @patch('engine.engine.pygame.event')
    def test_f3_toggles_hud(self, mock_event):
        f3_evt = MagicMock()
        f3_evt.type = pygame.KEYDOWN
        f3_evt.key = pygame.K_F3
        mock_event.get.return_value = [f3_evt]
        self.engine._show_hud = True
        self.engine._handle_events()
        self.assertFalse(self.engine._show_hud)

    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.mouse')
    def test_mouseclick_recaptures(self, mock_mouse, mock_event):
        click_evt = MagicMock()
        click_evt.type = pygame.MOUSEBUTTONDOWN
        mock_event.get.return_value = [click_evt]
        self.engine._mouse_captured = False
        self.engine._handle_events()
        self.assertTrue(self.engine._mouse_captured)

    @patch('engine.engine.pygame.event')
    def test_mouseclick_ignored_when_captured(self, mock_event):
        click_evt = MagicMock()
        click_evt.type = pygame.MOUSEBUTTONDOWN
        mock_event.get.return_value = [click_evt]
        self.engine._mouse_captured = True
        result = self.engine._handle_events()
        self.assertTrue(result)
        self.assertTrue(self.engine._mouse_captured)

    @patch('engine.engine.pygame.event')
    def test_no_events(self, mock_event):
        mock_event.get.return_value = []
        result = self.engine._handle_events()
        self.assertTrue(result)


class TestEngineProcessInput(unittest.TestCase):
    """Tests pour _process_input."""

    def setUp(self):
        self.engine = _create_engine_patched()

    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_process_input_captured(self, mock_mouse, mock_key):
        self.engine._mouse_captured = True
        mock_mouse.get_rel.return_value = (10, 5)
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        self.engine._process_input(0.016)
        mock_mouse.get_rel.assert_called_once()

    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_process_input_not_captured(self, mock_mouse, mock_key):
        self.engine._mouse_captured = False
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        self.engine._process_input(0.016)
        mock_mouse.get_rel.assert_not_called()


class TestEngineRender(unittest.TestCase):
    """Tests pour _render et ses sous-méthodes."""

    def setUp(self):
        self.engine = _create_engine_patched()
        self.engine._hud_font = MagicMock()
        text_surf = MagicMock()
        text_surf.get_width.return_value = 60
        text_surf.get_height.return_value = 16
        self.engine._hud_font.render.return_value = text_surf

    def test_render_solid_with_grid_and_hud(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        self.engine._meshes = [mesh]
        self.engine._model_matrices = [Mat4.identity()]
        self.engine._render_mode = 'solid'
        self.engine._show_grid = True
        self.engine._show_hud = True
        with patch('engine.engine.pygame.display'):
            self.engine._render()
        self.engine.renderer.render_mesh.assert_called()
        self.engine.renderer.render_grid.assert_called()
        self.engine.renderer.render_crosshair.assert_called()
        self.engine.renderer.present_overlay.assert_called()

    def test_render_wireframe_no_grid_no_hud(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        self.engine._meshes = [mesh]
        self.engine._model_matrices = [Mat4.identity()]
        self.engine._render_mode = 'wireframe'
        self.engine._show_grid = False
        self.engine._show_hud = False
        with patch('engine.engine.pygame.display'):
            self.engine._render()
        self.engine.renderer.render_wireframe.assert_called()
        self.engine.renderer.render_grid.assert_not_called()

    def test_render_empty_scene(self):
        self.engine._meshes = []
        self.engine._model_matrices = []
        self.engine._show_grid = True
        self.engine._show_hud = False
        with patch('engine.engine.pygame.display'):
            self.engine._render()
        self.engine.renderer.clear.assert_called()


class TestEngineRenderFPS(unittest.TestCase):
    """Tests pour _render_fps."""

    def setUp(self):
        self.engine = _create_engine_patched()

    @patch('engine.engine.pygame.font')
    def test_render_fps_creates_font(self, mock_font_mod):
        text_surf = MagicMock()
        text_surf.get_width.return_value = 50
        text_surf.get_height.return_value = 14
        font_obj = MagicMock()
        font_obj.render.return_value = text_surf
        mock_font_mod.SysFont.return_value = font_obj
        self.engine._hud_font = None
        self.engine._render_fps()
        mock_font_mod.SysFont.assert_called_once()
        self.assertIsNotNone(self.engine._hud_font)

    def test_render_fps_reuses_font(self):
        font_obj = MagicMock()
        text_surf = MagicMock()
        text_surf.get_width.return_value = 50
        text_surf.get_height.return_value = 14
        font_obj.render.return_value = text_surf
        self.engine._hud_font = font_obj
        self.engine._render_fps()
        font_obj.render.assert_called()


class TestEngineRenderHUD(unittest.TestCase):
    """Tests pour _render_hud."""

    def setUp(self):
        self.engine = _create_engine_patched()

    @patch('engine.engine.pygame.font')
    def test_render_hud_creates_font(self, mock_font_mod):
        text_surf = MagicMock()
        text_surf.get_width.return_value = 100
        text_surf.get_height.return_value = 14
        font_obj = MagicMock()
        font_obj.render.return_value = text_surf
        mock_font_mod.SysFont.return_value = font_obj
        self.engine._hud_font = None
        self.engine._render_hud()
        mock_font_mod.SysFont.assert_called_once()

    def test_render_hud_displays_lines(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        self.engine._meshes = [Mesh(verts, faces)]
        font_obj = MagicMock()
        text_surf = MagicMock()
        text_surf.get_width.return_value = 200
        text_surf.get_height.return_value = 14
        font_obj.render.return_value = text_surf
        self.engine._hud_font = font_obj
        self.engine._render_hud()
        self.assertEqual(font_obj.render.call_count, 5)


class TestEngineRun(unittest.TestCase):
    """Tests pour la boucle principale run."""

    def setUp(self):
        self.engine = _create_engine_patched()
        self.engine._hud_font = MagicMock()
        text_surf = MagicMock()
        text_surf.get_width.return_value = 60
        text_surf.get_height.return_value = 16
        self.engine._hud_font.render.return_value = text_surf

    @patch('engine.engine.pygame.quit')
    @patch('engine.engine.pygame.display')
    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_run_quits_on_quit_event(self, mock_mouse, mock_key, mock_event,
                                     mock_display, mock_quit):
        quit_evt = MagicMock()
        quit_evt.type = pygame.QUIT
        mock_event.get.return_value = [quit_evt]
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        mock_mouse.get_rel.return_value = (0, 0)
        self.engine.run()
        self.assertFalse(self.engine._running)

    @patch('engine.engine.pygame.quit')
    @patch('engine.engine.pygame.display')
    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_run_caps_dt(self, mock_mouse, mock_key, mock_event,
                         mock_display, mock_quit):
        call_count = [0]
        quit_evt = MagicMock()
        quit_evt.type = pygame.QUIT

        def side_effect():
            call_count[0] += 1
            if call_count[0] >= 2:
                return [quit_evt]
            return []

        mock_event.get.side_effect = side_effect
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        mock_mouse.get_rel.return_value = (0, 0)
        self.engine.run()
        mock_quit.assert_called_once()


class TestEngineObjectManagement(unittest.TestCase):
    """Tests pour add_object, remove_object, get_object."""

    def setUp(self):
        self.engine = _create_engine_patched()

    def test_add_object(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        obj = self.engine.add_object("test", mesh)
        self.assertEqual(len(self.engine.objects), 1)
        self.assertEqual(obj.name, "test")

    def test_add_object_with_params(self):
        from engine.mesh import Mesh
        from engine.math3d import Vec3
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        obj = self.engine.add_object(
            "colored", mesh,
            position=Vec3(1.0, 2.0, 3.0),
            color=(1.0, 0.0, 0.0),
        )
        self.assertEqual(obj.color, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(obj.transform.position.x, 1.0)

    def test_remove_object(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        obj = self.engine.add_object("to_remove", mesh)
        self.engine.remove_object(obj)
        self.assertEqual(len(self.engine.objects), 0)

    def test_remove_nonexistent(self):
        from engine.mesh import Mesh
        from engine.scene import SceneObject
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        fake = SceneObject(mesh=mesh, name="fake")
        self.engine.remove_object(fake)
        self.assertEqual(len(self.engine.objects), 0)

    def test_get_object_found(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        self.engine.add_object("findme", mesh)
        found = self.engine.get_object("findme")
        self.assertIsNotNone(found)
        self.assertEqual(found.name, "findme")

    def test_get_object_not_found(self):
        found = self.engine.get_object("nope")
        self.assertIsNone(found)

    def test_objects_property(self):
        self.assertIsInstance(self.engine.objects, list)
        self.assertEqual(len(self.engine.objects), 0)


class TestEngineReset(unittest.TestCase):
    """Tests pour reset."""

    def setUp(self):
        self.engine = _create_engine_patched()

    def test_reset_clears_objects(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        self.engine.add_object("obj1", mesh)
        self.engine.add_mesh(mesh)
        self.engine.reset()
        self.assertEqual(len(self.engine.objects), 0)
        self.assertEqual(len(self.engine._meshes), 0)

    def test_reset_resets_camera(self):
        from engine.math3d import Vec3
        self.engine.camera.position = Vec3(100.0, 200.0, 300.0)
        self.engine.reset()
        self.assertAlmostEqual(self.engine.camera.position.x, 0.0, places=3)
        self.assertAlmostEqual(self.engine.camera.position.y, 5.0, places=3)
        self.assertAlmostEqual(self.engine.camera.position.z, 15.0, places=3)


class TestEngineStep(unittest.TestCase):
    """Tests pour step."""

    def setUp(self):
        self.engine = _create_engine_patched()
        self.engine._hud_font = MagicMock()
        text_surf = MagicMock()
        text_surf.get_width.return_value = 60
        text_surf.get_height.return_value = 16
        self.engine._hud_font.render.return_value = text_surf

    @patch('engine.engine.pygame.display')
    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_step_returns_true_normally(self, mock_mouse, mock_key,
                                        mock_event, mock_display):
        mock_event.get.return_value = []
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        mock_mouse.get_rel.return_value = (0, 0)
        result = self.engine.step(dt=0.016)
        self.assertTrue(result)

    @patch('engine.engine.pygame.display')
    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_step_returns_false_on_quit(self, mock_mouse, mock_key,
                                        mock_event, mock_display):
        quit_evt = MagicMock()
        quit_evt.type = pygame.QUIT
        mock_event.get.return_value = [quit_evt]
        result = self.engine.step(dt=0.016)
        self.assertFalse(result)

    @patch('engine.engine.pygame.display')
    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_step_fixed_dt(self, mock_mouse, mock_key, mock_event, mock_display):
        mock_event.get.return_value = []
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        mock_mouse.get_rel.return_value = (0, 0)
        result = self.engine.step(dt=1.0 / 60.0)
        self.assertTrue(result)

    @patch('engine.engine.pygame.display')
    @patch('engine.engine.pygame.event')
    @patch('engine.engine.pygame.key')
    @patch('engine.engine.pygame.mouse')
    def test_step_auto_dt(self, mock_mouse, mock_key, mock_event, mock_display):
        mock_event.get.return_value = []
        pressed = MagicMock()
        pressed.__getitem__ = MagicMock(return_value=False)
        mock_key.get_pressed.return_value = pressed
        mock_mouse.get_rel.return_value = (0, 0)
        result = self.engine.step()
        self.assertTrue(result)


class TestEngineRenderObjects(unittest.TestCase):
    """Tests pour le rendu des SceneObjects."""

    def setUp(self):
        self.engine = _create_engine_patched()
        self.engine._hud_font = MagicMock()
        text_surf = MagicMock()
        text_surf.get_width.return_value = 60
        text_surf.get_height.return_value = 16
        self.engine._hud_font.render.return_value = text_surf

    def test_render_active_objects(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        self.engine.add_object("obj", mesh, color=(1.0, 0.0, 0.0))
        self.engine._render_mode = 'solid'
        self.engine._show_grid = False
        self.engine._show_hud = False
        with patch('engine.engine.pygame.display'):
            self.engine._render()
        self.engine.renderer.render_mesh.assert_called()

    def test_inactive_objects_not_rendered(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        obj = self.engine.add_object("hidden", mesh)
        obj.active = False
        self.engine._meshes = []
        self.engine._model_matrices = []
        self.engine._render_mode = 'solid'
        self.engine._show_grid = False
        self.engine._show_hud = False
        with patch('engine.engine.pygame.display'):
            self.engine._render()
        self.engine.renderer.render_mesh.assert_not_called()


if __name__ == '__main__':
    unittest.main()
