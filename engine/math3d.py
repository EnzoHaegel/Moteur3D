import numpy as np
from functools import lru_cache


class Vec3:
    """Vecteur 3D optimisé avec NumPy."""

    __slots__ = ('_data',)

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        """Initialise un vecteur 3D avec les composantes x, y, z."""
        self._data = np.array([x, y, z], dtype=np.float32)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Vec3':
        """Crée un Vec3 à partir d'un tableau NumPy existant."""
        v = cls.__new__(cls)
        v._data = arr.astype(np.float32)
        return v

    @property
    def x(self) -> float:
        """Composante X du vecteur."""
        return float(self._data[0])

    @property
    def y(self) -> float:
        """Composante Y du vecteur."""
        return float(self._data[1])

    @property
    def z(self) -> float:
        """Composante Z du vecteur."""
        return float(self._data[2])

    @x.setter
    def x(self, val: float):
        self._data[0] = val

    @y.setter
    def y(self, val: float):
        self._data[1] = val

    @z.setter
    def z(self, val: float):
        self._data[2] = val

    def __add__(self, other: 'Vec3') -> 'Vec3':
        return Vec3.from_array(self._data + other._data)

    def __sub__(self, other: 'Vec3') -> 'Vec3':
        return Vec3.from_array(self._data - other._data)

    def __mul__(self, scalar: float) -> 'Vec3':
        return Vec3.from_array(self._data * scalar)

    def __rmul__(self, scalar: float) -> 'Vec3':
        return self.__mul__(scalar)

    def __neg__(self) -> 'Vec3':
        return Vec3.from_array(-self._data)

    def __repr__(self) -> str:
        return f"Vec3({self.x:.4f}, {self.y:.4f}, {self.z:.4f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return np.allclose(self._data, other._data, atol=1e-6)

    def dot(self, other: 'Vec3') -> float:
        """Produit scalaire entre deux vecteurs."""
        return float(np.dot(self._data, other._data))

    def cross(self, other: 'Vec3') -> 'Vec3':
        """Produit vectoriel entre deux vecteurs."""
        return Vec3.from_array(np.cross(self._data, other._data))

    def length(self) -> float:
        """Norme (longueur) du vecteur."""
        return float(np.linalg.norm(self._data))

    def length_squared(self) -> float:
        """Carré de la norme du vecteur (évite la racine carrée)."""
        return float(np.dot(self._data, self._data))

    def normalized(self) -> 'Vec3':
        """Retourne le vecteur unitaire (normalisé)."""
        n = np.linalg.norm(self._data)
        if n < 1e-8:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3.from_array(self._data / n)

    def to_array(self) -> np.ndarray:
        """Retourne le tableau NumPy sous-jacent."""
        return self._data.copy()

    def to_vec4(self, w: float = 1.0) -> np.ndarray:
        """Convertit en coordonnées homogènes (vec4)."""
        return np.array([self._data[0], self._data[1], self._data[2], w], dtype=np.float32)


class Mat4:
    """Matrice 4x4 optimisée avec NumPy pour les transformations 3D."""

    __slots__ = ('_data',)

    def __init__(self, data: np.ndarray = None):
        """Initialise une matrice 4x4. Par défaut, matrice identité."""
        if data is not None:
            self._data = data.astype(np.float32)
        else:
            self._data = np.eye(4, dtype=np.float32)

    @staticmethod
    def identity() -> 'Mat4':
        """Retourne la matrice identité 4x4."""
        return Mat4()

    @staticmethod
    def translation(tx: float, ty: float, tz: float) -> 'Mat4':
        """Crée une matrice de translation."""
        m = np.eye(4, dtype=np.float32)
        m[0, 3] = tx
        m[1, 3] = ty
        m[2, 3] = tz
        return Mat4(m)

    @staticmethod
    def scale(sx: float, sy: float, sz: float) -> 'Mat4':
        """Crée une matrice de mise à l'échelle."""
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = sx
        m[1, 1] = sy
        m[2, 2] = sz
        m[3, 3] = 1.0
        return Mat4(m)

    @staticmethod
    def rotation_x(angle_rad: float) -> 'Mat4':
        """Crée une matrice de rotation autour de l'axe X."""
        c = np.float32(np.cos(angle_rad))
        s = np.float32(np.sin(angle_rad))
        m = np.eye(4, dtype=np.float32)
        m[1, 1] = c
        m[1, 2] = -s
        m[2, 1] = s
        m[2, 2] = c
        return Mat4(m)

    @staticmethod
    def rotation_y(angle_rad: float) -> 'Mat4':
        """Crée une matrice de rotation autour de l'axe Y."""
        c = np.float32(np.cos(angle_rad))
        s = np.float32(np.sin(angle_rad))
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 2] = s
        m[2, 0] = -s
        m[2, 2] = c
        return Mat4(m)

    @staticmethod
    def rotation_z(angle_rad: float) -> 'Mat4':
        """Crée une matrice de rotation autour de l'axe Z."""
        c = np.float32(np.cos(angle_rad))
        s = np.float32(np.sin(angle_rad))
        m = np.eye(4, dtype=np.float32)
        m[0, 0] = c
        m[0, 1] = -s
        m[1, 0] = s
        m[1, 1] = c
        return Mat4(m)

    @staticmethod
    def perspective(fov_rad: float, aspect: float, near: float, far: float) -> 'Mat4':
        """Crée une matrice de projection perspective."""
        f = 1.0 / np.tan(fov_rad / 2.0)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2.0 * far * near) / (near - far)
        m[3, 2] = -1.0
        return Mat4(m)

    @staticmethod
    def look_at(eye: Vec3, target: Vec3, up: Vec3) -> 'Mat4':
        """Crée une matrice de vue (caméra) look-at."""
        forward = (target - eye).normalized()
        right = forward.cross(up).normalized()
        cam_up = right.cross(forward)

        m = np.eye(4, dtype=np.float32)
        m[0, 0] = right.x
        m[0, 1] = right.y
        m[0, 2] = right.z
        m[0, 3] = -right.dot(eye)
        m[1, 0] = cam_up.x
        m[1, 1] = cam_up.y
        m[1, 2] = cam_up.z
        m[1, 3] = -cam_up.dot(eye)
        m[2, 0] = -forward.x
        m[2, 1] = -forward.y
        m[2, 2] = -forward.z
        m[2, 3] = forward.dot(eye)
        return Mat4(m)

    def __matmul__(self, other: 'Mat4') -> 'Mat4':
        """Multiplication matricielle avec l'opérateur @."""
        return Mat4(self._data @ other._data)

    def transform_point(self, v: Vec3) -> Vec3:
        """Transforme un point 3D par la matrice (avec perspective divide)."""
        p = self._data @ v.to_vec4(1.0)
        if abs(p[3]) > 1e-8:
            p /= p[3]
        return Vec3(float(p[0]), float(p[1]), float(p[2]))

    def transform_vec4(self, v4: np.ndarray) -> np.ndarray:
        """Transforme un vecteur homogène 4D."""
        return self._data @ v4

    def transform_points_batch(self, points: np.ndarray) -> np.ndarray:
        """Transforme un batch de points (Nx3) de manière vectorisée."""
        n = points.shape[0]
        ones = np.ones((n, 1), dtype=np.float32)
        homogeneous = np.hstack([points, ones])
        transformed = (self._data @ homogeneous.T).T
        w = transformed[:, 3:4]
        w = np.where(np.abs(w) < 1e-8, 1.0, w)
        return transformed[:, :3] / w

    def transform_points_batch_with_w(self, points: np.ndarray) -> tuple:
        """Transforme un batch de points et retourne aussi le w clip-space.

        Returns:
            Tuple (ndc_coords Nx3, w_values N) pour le near-plane clipping.
        """
        n = points.shape[0]
        ones = np.ones((n, 1), dtype=np.float32)
        homogeneous = np.hstack([points, ones])
        transformed = (self._data @ homogeneous.T).T
        w_raw = transformed[:, 3].copy()
        w = transformed[:, 3:4]
        w = np.where(np.abs(w) < 1e-8, 1.0, w)
        return transformed[:, :3] / w, w_raw

    @property
    def data(self) -> np.ndarray:
        """Accès au tableau NumPy sous-jacent."""
        return self._data

    def __repr__(self) -> str:
        return f"Mat4(\n{self._data}\n)"
