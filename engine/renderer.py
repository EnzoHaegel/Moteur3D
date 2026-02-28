import numpy as np
import pygame
import ctypes
from OpenGL.GL import *
from .math3d import Mat4
from .mesh import Mesh


_MESH_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_normal;

void main() {
    gl_Position = u_mvp * vec4(aPos, 1.0);
    v_normal = mat3(u_model) * aNormal;
}
"""

_MESH_FRAG = """
#version 330 core
in vec3 v_normal;

uniform vec3 u_lightDir;
uniform float u_ambient;
uniform vec3 u_baseColor;

out vec4 FragColor;

void main() {
    vec3 n = normalize(v_normal);
    float diff = max(dot(n, u_lightDir), 0.0);
    float intensity = clamp(u_ambient + diff * (1.0 - u_ambient), 0.0, 1.0);
    FragColor = vec4(u_baseColor * intensity, 1.0);
}
"""

_LINE_VERT = """
#version 330 core
layout(location = 0) in vec3 aPos;

uniform mat4 u_mvp;

void main() {
    gl_Position = u_mvp * vec4(aPos, 1.0);
}
"""

_LINE_FRAG = """
#version 330 core
uniform vec3 u_color;

out vec4 FragColor;

void main() {
    FragColor = vec4(u_color, 1.0);
}
"""

_OVERLAY_VERT = """
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;

out vec2 v_uv;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    v_uv = aUV;
}
"""

_OVERLAY_FRAG = """
#version 330 core
in vec2 v_uv;

uniform sampler2D u_texture;

out vec4 FragColor;

void main() {
    FragColor = texture(u_texture, v_uv);
}
"""


class _MeshGPU:
    """Données GPU d'un maillage uploadé (VAO, VBOs, EBO)."""

    __slots__ = ('vao', 'vbo', 'nbo', 'ebo', 'count')

    def __init__(self, vao, vbo, nbo, ebo, count):
        self.vao = vao
        self.vbo = vbo
        self.nbo = nbo
        self.ebo = ebo
        self.count = count


class Renderer:
    """Pipeline de rendu 3D accéléré par GPU via OpenGL 3.3 core."""

    __slots__ = (
        '_width', '_height',
        '_mesh_prog', '_line_prog', '_overlay_prog',
        '_mu', '_lu', '_ou',
        '_mesh_cache',
        '_light_dir', '_ambient', '_base_color',
        '_grid_vao', '_grid_count',
        '_cross_vao',
        '_overlay_vao', '_overlay_tex', '_overlay_surface',
        '_identity', '_grid_color', '_white',
    )

    def __init__(self, width: int, height: int):
        """Initialise le renderer OpenGL après création du contexte.

        Args:
            width: Largeur de l'écran en pixels.
            height: Hauteur de l'écran en pixels.
        """
        self._width = width
        self._height = height

        glClearColor(30.0 / 255.0, 30.0 / 255.0, 45.0 / 255.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glViewport(0, 0, width, height)

        self._mesh_prog = self._build(_MESH_VERT, _MESH_FRAG)
        self._line_prog = self._build(_LINE_VERT, _LINE_FRAG)
        self._overlay_prog = self._build(_OVERLAY_VERT, _OVERLAY_FRAG)

        self._mu = {
            n: glGetUniformLocation(self._mesh_prog, f"u_{n}")
            for n in ('mvp', 'model', 'lightDir', 'ambient', 'baseColor')
        }
        self._lu = {
            n: glGetUniformLocation(self._line_prog, f"u_{n}")
            for n in ('mvp', 'color')
        }
        self._ou = {
            'texture': glGetUniformLocation(self._overlay_prog, 'u_texture'),
        }

        self._mesh_cache: dict[int, _MeshGPU] = {}

        light = np.array([0.5, 0.8, 0.3], dtype=np.float32)
        self._light_dir = (light / np.linalg.norm(light)).copy()
        self._ambient = np.float32(0.15)
        self._base_color = np.array(
            [180.0 / 255.0, 160.0 / 255.0, 140.0 / 255.0], dtype=np.float32)

        self._identity = np.eye(4, dtype=np.float32)
        self._grid_color = np.array(
            [60.0 / 255.0, 60.0 / 255.0, 80.0 / 255.0], dtype=np.float32)
        self._white = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        self._grid_vao = 0
        self._grid_count = 0
        self._init_grid()

        self._cross_vao = 0
        self._init_crosshair()

        self._overlay_surface = pygame.Surface(
            (width, height), pygame.SRCALPHA)
        self._overlay_tex = glGenTextures(1)
        self._overlay_vao = 0
        self._init_overlay()

    @property
    def width(self) -> int:
        """Largeur de l'écran."""
        return self._width

    @property
    def height(self) -> int:
        """Hauteur de l'écran."""
        return self._height

    @property
    def overlay(self) -> pygame.Surface:
        """Surface Pygame transparente pour le HUD 2D."""
        return self._overlay_surface

    @staticmethod
    def _compile(source: str, stage: int) -> int:
        """Compile un shader GLSL et retourne son identifiant."""
        s = glCreateShader(stage)
        glShaderSource(s, source)
        glCompileShader(s)
        if glGetShaderiv(s, GL_COMPILE_STATUS) != GL_TRUE:
            log = glGetShaderInfoLog(s).decode()
            glDeleteShader(s)
            raise RuntimeError(f"Shader compile: {log}")
        return s

    @classmethod
    def _build(cls, vert_src: str, frag_src: str) -> int:
        """Compile et lie un programme shader (vertex + fragment)."""
        vs = cls._compile(vert_src, GL_VERTEX_SHADER)
        fs = cls._compile(frag_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        if glGetProgramiv(prog, GL_LINK_STATUS) != GL_TRUE:
            log = glGetProgramInfoLog(prog).decode()
            glDeleteProgram(prog)
            raise RuntimeError(f"Program link: {log}")
        glDeleteShader(vs)
        glDeleteShader(fs)
        return prog

    def _get_gpu(self, mesh: Mesh) -> _MeshGPU:
        """Retourne les données GPU d'un maillage (upload au premier appel)."""
        key = id(mesh)
        gpu = self._mesh_cache.get(key)
        if gpu is None:
            gpu = self._upload(mesh)
            self._mesh_cache[key] = gpu
        return gpu

    @staticmethod
    def _upload(mesh: Mesh) -> _MeshGPU:
        """Upload les vertices, normales et indices d'un maillage sur le GPU."""
        verts = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        norms = np.ascontiguousarray(mesh.normals, dtype=np.float32)
        idx = np.ascontiguousarray(mesh.faces.ravel(), dtype=np.uint32)

        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        nbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, nbo)
        glBufferData(GL_ARRAY_BUFFER, norms.nbytes, norms, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)

        glBindVertexArray(0)

        return _MeshGPU(vao, vbo, nbo, ebo, len(idx))

    def _init_grid(self, size: int = 20, spacing: float = 2.0):
        """Prépare le VAO de la grille au sol."""
        half = size * spacing
        pts = []
        for i in range(-size, size + 1):
            p = i * spacing
            pts.extend([[p, 0.0, -half], [p, 0.0, half]])
            pts.extend([[-half, 0.0, p], [half, 0.0, p]])
        data = np.array(pts, dtype=np.float32)
        self._grid_count = len(pts)

        self._grid_vao = glGenVertexArrays(1)
        glBindVertexArray(self._grid_vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _init_crosshair(self):
        """Prépare le VAO du réticule au centre de l'écran."""
        sx = 10.0 / (self._width * 0.5)
        sy = 10.0 / (self._height * 0.5)
        data = np.array([
            [-sx, 0.0, 0.0], [sx, 0.0, 0.0],
            [0.0, -sy, 0.0], [0.0, sy, 0.0],
        ], dtype=np.float32)

        self._cross_vao = glGenVertexArrays(1)
        glBindVertexArray(self._cross_vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, data.nbytes, data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

    def _init_overlay(self):
        """Prépare le quad plein-écran et la texture pour l'overlay HUD."""
        quad = np.array([
            -1.0, -1.0, 0.0, 0.0,
            1.0, -1.0, 1.0, 0.0,
            -1.0,  1.0, 0.0, 1.0,
            1.0,  1.0, 1.0, 1.0,
        ], dtype=np.float32)

        self._overlay_vao = glGenVertexArrays(1)
        glBindVertexArray(self._overlay_vao)
        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STATIC_DRAW)
        stride = 4 * 4
        glVertexAttribPointer(
            0, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(
            1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        glBindTexture(GL_TEXTURE_2D, self._overlay_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            self._width, self._height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, None,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def clear(self):
        """Efface l'écran 3D et la surface overlay."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        self._overlay_surface.fill((0, 0, 0, 0))

    def render_mesh(self, mesh: Mesh, mvp: Mat4, model: Mat4 = None):
        """Rend un maillage avec éclairage directionnel via le GPU.

        Args:
            mesh: Le maillage à rendre.
            mvp: Matrice Model-View-Projection combinée.
            model: Matrice modèle pour transformer les normales (optionnel).
        """
        gpu = self._get_gpu(mesh)
        glUseProgram(self._mesh_prog)
        glUniformMatrix4fv(self._mu['mvp'], 1, GL_TRUE, mvp.data)
        m = model.data if model is not None else self._identity
        glUniformMatrix4fv(self._mu['model'], 1, GL_TRUE, m)
        glUniform3fv(self._mu['lightDir'], 1, self._light_dir)
        glUniform1f(self._mu['ambient'], self._ambient)
        glUniform3fv(self._mu['baseColor'], 1, self._base_color)
        glBindVertexArray(gpu.vao)
        glDrawElements(GL_TRIANGLES, gpu.count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

    def render_wireframe(self, mesh: Mesh, mvp: Mat4, color=(0, 255, 100)):
        """Rend un maillage en mode fil de fer.

        Args:
            mesh: Le maillage à rendre.
            mvp: Matrice Model-View-Projection combinée.
            color: Couleur RGB des lignes.
        """
        gpu = self._get_gpu(mesh)
        glUseProgram(self._line_prog)
        glUniformMatrix4fv(self._lu['mvp'], 1, GL_TRUE, mvp.data)
        c = np.array(
            [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0],
            dtype=np.float32,
        )
        glUniform3fv(self._lu['color'], 1, c)
        glDisable(GL_CULL_FACE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBindVertexArray(gpu.vao)
        glDrawElements(GL_TRIANGLES, gpu.count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glEnable(GL_CULL_FACE)

    def render_grid(self, vp: Mat4):
        """Rend la grille au sol.

        Args:
            vp: Matrice View-Projection.
        """
        glUseProgram(self._line_prog)
        glUniformMatrix4fv(self._lu['mvp'], 1, GL_TRUE, vp.data)
        glUniform3fv(self._lu['color'], 1, self._grid_color)
        glBindVertexArray(self._grid_vao)
        glDrawArrays(GL_LINES, 0, self._grid_count)
        glBindVertexArray(0)

    def render_crosshair(self):
        """Dessine un réticule (crosshair) au centre de l'écran."""
        glUseProgram(self._line_prog)
        glUniformMatrix4fv(self._lu['mvp'], 1, GL_TRUE, self._identity)
        glUniform3fv(self._lu['color'], 1, self._white)
        glDisable(GL_DEPTH_TEST)
        glBindVertexArray(self._cross_vao)
        glDrawArrays(GL_LINES, 0, 4)
        glBindVertexArray(0)
        glEnable(GL_DEPTH_TEST)

    def present_overlay(self):
        """Upload la surface HUD sur le GPU et l'affiche par-dessus la scène."""
        raw = pygame.image.tostring(self._overlay_surface, 'RGBA', True)
        glBindTexture(GL_TEXTURE_2D, self._overlay_tex)
        glTexSubImage2D(
            GL_TEXTURE_2D, 0, 0, 0,
            self._width, self._height,
            GL_RGBA, GL_UNSIGNED_BYTE, raw,
        )
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glUseProgram(self._overlay_prog)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._overlay_tex)
        glUniform1i(self._ou['texture'], 0)
        glBindVertexArray(self._overlay_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
