import numpy as np
import math
from .math3d import Vec3, Mat4


class Camera:
    """Caméra FPS style Minecraft avec contrôle ZQSD + souris."""

    __slots__ = (
        '_position', '_yaw', '_pitch', '_speed', '_sensitivity',
        '_fov', '_aspect', '_near', '_far',
        '_forward', '_right', '_up', '_view_dirty', '_proj_dirty',
        '_view_matrix', '_proj_matrix',
    )

    def __init__(
        self,
        position: Vec3 = None,
        yaw: float = -90.0,
        pitch: float = 0.0,
        fov: float = 70.0,
        aspect: float = 16 / 9,
        near: float = 0.1,
        far: float = 1000.0,
        speed: float = 10.0,
        sensitivity: float = 0.1,
    ):
        """Initialise la caméra avec position, orientation et paramètres de projection."""
        self._position = position if position else Vec3(0.0, 5.0, 10.0)
        self._yaw = yaw
        self._pitch = pitch
        self._speed = speed
        self._sensitivity = sensitivity
        self._fov = fov
        self._aspect = aspect
        self._near = near
        self._far = far
        self._forward = Vec3(0.0, 0.0, -1.0)
        self._right = Vec3(1.0, 0.0, 0.0)
        self._up = Vec3(0.0, 1.0, 0.0)
        self._view_dirty = True
        self._proj_dirty = True
        self._view_matrix = Mat4.identity()
        self._proj_matrix = Mat4.identity()
        self._update_vectors()

    @property
    def position(self) -> Vec3:
        """Position actuelle de la caméra."""
        return self._position

    @position.setter
    def position(self, value: Vec3):
        self._position = value
        self._view_dirty = True

    @property
    def yaw(self) -> float:
        """Angle de lacet (rotation horizontale) en degrés."""
        return self._yaw

    @property
    def pitch(self) -> float:
        """Angle de tangage (rotation verticale) en degrés."""
        return self._pitch

    @property
    def forward(self) -> Vec3:
        """Vecteur direction avant de la caméra."""
        return self._forward

    @property
    def right(self) -> Vec3:
        """Vecteur direction droite de la caméra."""
        return self._right

    @property
    def up(self) -> Vec3:
        """Vecteur direction haut de la caméra."""
        return self._up

    @property
    def speed(self) -> float:
        """Vitesse de déplacement de la caméra."""
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = value

    @property
    def fov(self) -> float:
        """Champ de vision en degrés."""
        return self._fov

    @fov.setter
    def fov(self, value: float):
        self._fov = value
        self._proj_dirty = True

    @property
    def aspect(self) -> float:
        """Ratio d'aspect de la projection."""
        return self._aspect

    @aspect.setter
    def aspect(self, value: float):
        self._aspect = value
        self._proj_dirty = True

    def _update_vectors(self):
        """Recalcule les vecteurs de direction à partir de yaw et pitch."""
        yaw_rad = math.radians(self._yaw)
        pitch_rad = math.radians(self._pitch)

        cos_pitch = math.cos(pitch_rad)
        self._forward = Vec3(
            math.cos(yaw_rad) * cos_pitch,
            math.sin(pitch_rad),
            math.sin(yaw_rad) * cos_pitch,
        ).normalized()

        world_up = Vec3(0.0, 1.0, 0.0)
        self._right = self._forward.cross(world_up).normalized()
        self._up = self._right.cross(self._forward).normalized()
        self._view_dirty = True

    def process_mouse(self, dx: float, dy: float):
        """Traite le mouvement de la souris pour orienter la caméra."""
        self._yaw += dx * self._sensitivity
        self._pitch -= dy * self._sensitivity
        self._pitch = max(-89.0, min(89.0, self._pitch))
        self._update_vectors()

    def process_keyboard(self, keys: dict, dt: float):
        """Traite les entrées clavier ZQSD pour déplacer la caméra.

        Args:
            keys: Dictionnaire des touches pressées ('z', 'q', 's', 'd', 'space', 'shift').
            dt: Delta time en secondes.
        """
        velocity = self._speed * dt

        forward_flat = Vec3(self._forward.x, 0.0, self._forward.z).normalized()
        right_flat = Vec3(self._right.x, 0.0, self._right.z).normalized()

        move = Vec3(0.0, 0.0, 0.0)

        if keys.get('z', False):
            move = move + forward_flat
        if keys.get('s', False):
            move = move - forward_flat
        if keys.get('d', False):
            move = move + right_flat
        if keys.get('q', False):
            move = move - right_flat
        if keys.get('space', False):
            move = move + Vec3(0.0, 1.0, 0.0)
        if keys.get('shift', False):
            move = move - Vec3(0.0, 1.0, 0.0)

        if move.length_squared() > 0.0001:
            move = move.normalized() * velocity
            self._position = self._position + move
            self._view_dirty = True

    def get_view_matrix(self) -> Mat4:
        """Retourne la matrice de vue (mise en cache si non modifiée)."""
        if self._view_dirty:
            target = self._position + self._forward
            self._view_matrix = Mat4.look_at(
                self._position, target, Vec3(0.0, 1.0, 0.0))
            self._view_dirty = False
        return self._view_matrix

    def get_projection_matrix(self) -> Mat4:
        """Retourne la matrice de projection perspective (mise en cache si non modifiée)."""
        if self._proj_dirty:
            fov_rad = math.radians(self._fov)
            self._proj_matrix = Mat4.perspective(
                fov_rad, self._aspect, self._near, self._far)
            self._proj_dirty = False
        return self._proj_matrix

    def get_vp_matrix(self) -> Mat4:
        """Retourne la matrice View-Projection combinée."""
        return self.get_projection_matrix() @ self.get_view_matrix()
