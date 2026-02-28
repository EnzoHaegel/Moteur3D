import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def _make_gl_mocks():
    """Construit un dictionnaire de mocks pour toutes les fonctions OpenGL utilisées."""
    gl = MagicMock()
    gl.GL_VERTEX_SHADER = 0x8B31
    gl.GL_FRAGMENT_SHADER = 0x8B30
    gl.GL_COMPILE_STATUS = 0x8B81
    gl.GL_LINK_STATUS = 0x8B82
    gl.GL_TRUE = 1
    gl.GL_FALSE = 0
    gl.GL_DEPTH_TEST = 0x0B71
    gl.GL_LESS = 0x0201
    gl.GL_CULL_FACE = 0x0B44
    gl.GL_BACK = 0x0405
    gl.GL_CCW = 0x0901
    gl.GL_COLOR_BUFFER_BIT = 0x4000
    gl.GL_DEPTH_BUFFER_BIT = 0x0100
    gl.GL_TRIANGLES = 0x0004
    gl.GL_UNSIGNED_INT = 0x1405
    gl.GL_LINES = 0x0001
    gl.GL_TRIANGLE_STRIP = 0x0005
    gl.GL_FLOAT = 0x1406
    gl.GL_ARRAY_BUFFER = 0x8892
    gl.GL_ELEMENT_ARRAY_BUFFER = 0x8893
    gl.GL_STATIC_DRAW = 0x88E4
    gl.GL_LINE = 0x1B01
    gl.GL_FILL = 0x1B02
    gl.GL_FRONT_AND_BACK = 0x0408
    gl.GL_BLEND = 0x0BE2
    gl.GL_SRC_ALPHA = 0x0302
    gl.GL_ONE_MINUS_SRC_ALPHA = 0x0303
    gl.GL_TEXTURE_2D = 0x0DE1
    gl.GL_TEXTURE0 = 0x84C0
    gl.GL_TEXTURE_MIN_FILTER = 0x2801
    gl.GL_TEXTURE_MAG_FILTER = 0x2800
    gl.GL_LINEAR = 0x2601
    gl.GL_RGBA = 0x1908
    gl.GL_UNSIGNED_BYTE = 0x1401
    gl.glCreateShader = MagicMock(return_value=1)
    gl.glShaderSource = MagicMock()
    gl.glCompileShader = MagicMock()
    gl.glGetShaderiv = MagicMock(return_value=1)
    gl.glGetShaderInfoLog = MagicMock(return_value=b"error")
    gl.glDeleteShader = MagicMock()
    gl.glCreateProgram = MagicMock(return_value=10)
    gl.glAttachShader = MagicMock()
    gl.glLinkProgram = MagicMock()
    gl.glGetProgramiv = MagicMock(return_value=1)
    gl.glGetProgramInfoLog = MagicMock(return_value=b"link error")
    gl.glDeleteProgram = MagicMock()
    gl.glGetUniformLocation = MagicMock(return_value=0)
    gl.glUseProgram = MagicMock()
    gl.glUniformMatrix4fv = MagicMock()
    gl.glUniform3fv = MagicMock()
    gl.glUniform1f = MagicMock()
    gl.glUniform1i = MagicMock()
    gl.glClearColor = MagicMock()
    gl.glClear = MagicMock()
    gl.glEnable = MagicMock()
    gl.glDisable = MagicMock()
    gl.glDepthFunc = MagicMock()
    gl.glCullFace = MagicMock()
    gl.glFrontFace = MagicMock()
    gl.glViewport = MagicMock()
    gl.glGenVertexArrays = MagicMock(return_value=1)
    gl.glBindVertexArray = MagicMock()
    gl.glGenBuffers = MagicMock(return_value=1)
    gl.glBindBuffer = MagicMock()
    gl.glBufferData = MagicMock()
    gl.glVertexAttribPointer = MagicMock()
    gl.glEnableVertexAttribArray = MagicMock()
    gl.glDrawElements = MagicMock()
    gl.glDrawArrays = MagicMock()
    gl.glPolygonMode = MagicMock()
    gl.glGenTextures = MagicMock(return_value=1)
    gl.glBindTexture = MagicMock()
    gl.glTexParameteri = MagicMock()
    gl.glTexImage2D = MagicMock()
    gl.glTexSubImage2D = MagicMock()
    gl.glActiveTexture = MagicMock()
    gl.glBlendFunc = MagicMock()
    return gl


def _patch_gl(gl_mock):
    """Retourne un dict de patchs OpenGL.GL.* à utiliser avec patch.dict."""
    mapping = {}
    for name in dir(gl_mock):
        if name.startswith('gl') or name.startswith('GL_'):
            mapping[f'OpenGL.GL.{name}'] = getattr(gl_mock, name)
    return mapping


class TestMeshGPU(unittest.TestCase):
    """Tests pour la dataclass _MeshGPU."""

    def test_init(self):
        from engine.renderer import _MeshGPU
        gpu = _MeshGPU(vao=1, vbo=2, nbo=3, ebo=4, count=36)
        self.assertEqual(gpu.vao, 1)
        self.assertEqual(gpu.vbo, 2)
        self.assertEqual(gpu.nbo, 3)
        self.assertEqual(gpu.ebo, 4)
        self.assertEqual(gpu.count, 36)


class TestRendererWithContext(unittest.TestCase):
    """Tests du Renderer avec contexte OpenGL mocké."""

    def _make_renderer(self):
        """Crée un Renderer en mockant toutes les fonctions GL."""
        gl = _make_gl_mocks()
        with patch.dict('sys.modules', {'OpenGL': MagicMock(), 'OpenGL.GL': gl}):
            import importlib
            import engine.renderer as mod
            importlib.reload(mod)
            renderer = mod.Renderer(800, 600)
        return renderer, gl, mod

    def setUp(self):
        self.gl = _make_gl_mocks()
        self._patcher = patch(
            'engine.renderer.glClearColor', self.gl.glClearColor)

        import engine.renderer as mod
        self.mod = mod

        self.orig_funcs = {}
        for name in dir(self.gl):
            if name.startswith('gl') or name.startswith('GL_'):
                if hasattr(mod, name):
                    self.orig_funcs[name] = getattr(mod, name)
                    setattr(mod, name, getattr(self.gl, name))

        self.renderer = mod.Renderer(800, 600)

    def tearDown(self):
        for name, orig in self.orig_funcs.items():
            setattr(self.mod, name, orig)

    def test_width_property(self):
        self.assertEqual(self.renderer.width, 800)

    def test_height_property(self):
        self.assertEqual(self.renderer.height, 600)

    def test_overlay_is_surface(self):
        import pygame
        self.assertIsInstance(self.renderer.overlay, pygame.Surface)

    def test_overlay_size(self):
        overlay = self.renderer.overlay
        self.assertEqual(overlay.get_width(), 800)
        self.assertEqual(overlay.get_height(), 600)

    def test_clear(self):
        self.renderer.clear()
        self.gl.glClear.assert_called()

    def test_render_mesh_without_model(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        from engine.math3d import Mat4
        mvp = Mat4.identity()
        self.renderer.render_mesh(mesh, mvp)
        self.gl.glUseProgram.assert_called()
        self.gl.glDrawElements.assert_called()

    def test_render_mesh_with_model(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        mvp = Mat4.identity()
        model = Mat4.scale(2.0, 2.0, 2.0)
        self.renderer.render_mesh(mesh, mvp, model)
        self.gl.glDrawElements.assert_called()

    def test_render_mesh_cache(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        mvp = Mat4.identity()
        self.renderer.render_mesh(mesh, mvp)
        upload_count_1 = self.gl.glGenVertexArrays.call_count
        self.renderer.render_mesh(mesh, mvp)
        upload_count_2 = self.gl.glGenVertexArrays.call_count
        self.assertEqual(upload_count_1, upload_count_2)

    def test_render_wireframe(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        mvp = Mat4.identity()
        self.renderer.render_wireframe(mesh, mvp)
        self.gl.glPolygonMode.assert_called()
        self.gl.glDrawElements.assert_called()

    def test_render_wireframe_custom_color(self):
        from engine.mesh import Mesh
        from engine.math3d import Mat4
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        mvp = Mat4.identity()
        self.renderer.render_wireframe(mesh, mvp, color=(255, 0, 0))
        self.gl.glUniform3fv.assert_called()

    def test_render_grid(self):
        from engine.math3d import Mat4
        vp = Mat4.identity()
        self.renderer.render_grid(vp)
        self.gl.glDrawArrays.assert_called()

    def test_render_crosshair(self):
        self.renderer.render_crosshair()
        self.gl.glDrawArrays.assert_called()
        self.gl.glDisable.assert_called()
        self.gl.glEnable.assert_called()

    def test_present_overlay(self):
        self.renderer.present_overlay()
        self.gl.glTexSubImage2D.assert_called()
        self.gl.glDrawArrays.assert_called()
        self.gl.glEnable.assert_called()
        self.gl.glDisable.assert_called()

    def test_compile_shader_failure(self):
        self.gl.glGetShaderiv.return_value = 0
        self.gl.glGetShaderInfoLog.return_value = b"syntax error"
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.Renderer._compile("bad code", self.gl.GL_VERTEX_SHADER)
        self.assertIn("Shader compile", str(ctx.exception))
        self.gl.glDeleteShader.assert_called()

    def test_build_link_failure(self):
        self.gl.glGetShaderiv.return_value = 1
        self.gl.glGetProgramiv.return_value = 0
        self.gl.glGetProgramInfoLog.return_value = b"link fail"
        with self.assertRaises(RuntimeError) as ctx:
            self.mod.Renderer._build("vert", "frag")
        self.assertIn("Program link", str(ctx.exception))
        self.gl.glDeleteProgram.assert_called()

    def test_upload_static(self):
        from engine.mesh import Mesh
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = Mesh(verts, faces)
        gpu = self.mod.Renderer._upload(mesh)
        self.assertEqual(gpu.count, 3)
        self.gl.glGenVertexArrays.assert_called()
        self.gl.glBufferData.assert_called()


class TestRendererInitGrid(unittest.TestCase):
    """Vérifie que la grille est correctement initialisée."""

    def test_grid_count(self):
        gl = _make_gl_mocks()
        import engine.renderer as mod
        orig_funcs = {}
        for name in dir(gl):
            if name.startswith('gl') or name.startswith('GL_'):
                if hasattr(mod, name):
                    orig_funcs[name] = getattr(mod, name)
                    setattr(mod, name, getattr(gl, name))
        try:
            r = mod.Renderer(640, 480)
            expected = (20 * 2 + 1) * 4
            self.assertEqual(r._grid_count, expected)
        finally:
            for name, orig in orig_funcs.items():
                setattr(mod, name, orig)


if __name__ == '__main__':
    unittest.main()
