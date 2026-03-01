import math
import numpy as np
from .math3d import Vec3, Mat4


class Transform:
    """Transformation 3D composée d'une position, rotation (euler) et échelle."""

    __slots__ = (
        '_position', '_rotation', '_scale',
        '_matrix', '_dirty',
    )

    def __init__(
        self,
        position: Vec3 = None,
        rotation: Vec3 = None,
        scale: Vec3 = None,
    ):
        """Initialise une transformation 3D.

        Args:
            position: Position dans l'espace monde.
            rotation: Rotation en degrés (euler XYZ).
            scale: Facteurs d'échelle.
        """
        self._position = position if position else Vec3(0.0, 0.0, 0.0)
        self._rotation = rotation if rotation else Vec3(0.0, 0.0, 0.0)
        self._scale = scale if scale else Vec3(1.0, 1.0, 1.0)
        self._matrix = Mat4.identity()
        self._dirty = True

    @property
    def position(self) -> Vec3:
        """Position dans l'espace monde."""
        return self._position

    @position.setter
    def position(self, value: Vec3):
        self._position = value
        self._dirty = True

    @property
    def rotation(self) -> Vec3:
        """Rotation en degrés (euler XYZ)."""
        return self._rotation

    @rotation.setter
    def rotation(self, value: Vec3):
        self._rotation = value
        self._dirty = True

    @property
    def scale(self) -> Vec3:
        """Facteurs d'échelle."""
        return self._scale

    @scale.setter
    def scale(self, value: Vec3):
        self._scale = value
        self._dirty = True

    def get_model_matrix(self) -> Mat4:
        """Retourne la matrice modèle TRS (Translation * Rotation * Scale).

        Returns:
            Matrice 4x4 combinant translation, rotation et échelle.
        """
        if self._dirty:
            t = Mat4.translation(
                self._position.x, self._position.y, self._position.z)
            rx = Mat4.rotation_x(math.radians(self._rotation.x))
            ry = Mat4.rotation_y(math.radians(self._rotation.y))
            rz = Mat4.rotation_z(math.radians(self._rotation.z))
            s = Mat4.scale(self._scale.x, self._scale.y, self._scale.z)
            self._matrix = t @ ry @ rx @ rz @ s
            self._dirty = False
        return self._matrix

    def translate(self, offset: Vec3):
        """Déplace la transformation par un offset relatif.

        Args:
            offset: Vecteur de déplacement.
        """
        self._position = self._position + offset
        self._dirty = True

    def __repr__(self) -> str:
        return (
            f"Transform(pos={self._position}, "
            f"rot={self._rotation}, "
            f"scale={self._scale})"
        )
