import numpy as np
from .math3d import Vec3


class AABB:
    """Boîte englobante alignée sur les axes (Axis-Aligned Bounding Box)."""

    __slots__ = ('min_point', 'max_point')

    def __init__(self, min_point: Vec3, max_point: Vec3):
        """Initialise une AABB avec ses coins minimum et maximum.

        Args:
            min_point: Coin minimum (x_min, y_min, z_min).
            max_point: Coin maximum (x_max, y_max, z_max).
        """
        self.min_point = min_point
        self.max_point = max_point

    @staticmethod
    def from_mesh(mesh, transform=None) -> 'AABB':
        """Crée une AABB à partir d'un maillage, optionnellement transformé.

        Args:
            mesh: Le maillage source.
            transform: Transform optionnel pour calculer l'AABB en espace monde.

        Returns:
            AABB englobant le maillage.
        """
        if transform is not None:
            model = transform.get_model_matrix()
            pts = model.transform_points_batch(mesh.vertices)
        else:
            pts = mesh.vertices

        min_vals = pts.min(axis=0)
        max_vals = pts.max(axis=0)
        return AABB(
            Vec3(float(min_vals[0]), float(min_vals[1]), float(min_vals[2])),
            Vec3(float(max_vals[0]), float(max_vals[1]), float(max_vals[2])),
        )

    def intersects(self, other: 'AABB') -> bool:
        """Teste l'intersection avec une autre AABB.

        Args:
            other: L'autre boîte englobante.

        Returns:
            True si les deux AABB se chevauchent.
        """
        return (
            self.min_point.x <= other.max_point.x
            and self.max_point.x >= other.min_point.x
            and self.min_point.y <= other.max_point.y
            and self.max_point.y >= other.min_point.y
            and self.min_point.z <= other.max_point.z
            and self.max_point.z >= other.min_point.z
        )

    def contains_point(self, point: Vec3) -> bool:
        """Teste si un point est contenu dans l'AABB.

        Args:
            point: Le point à tester.

        Returns:
            True si le point est à l'intérieur de la boîte.
        """
        return (
            self.min_point.x <= point.x <= self.max_point.x
            and self.min_point.y <= point.y <= self.max_point.y
            and self.min_point.z <= point.z <= self.max_point.z
        )

    def center(self) -> Vec3:
        """Retourne le centre de l'AABB.

        Returns:
            Point central de la boîte.
        """
        return Vec3(
            (self.min_point.x + self.max_point.x) * 0.5,
            (self.min_point.y + self.max_point.y) * 0.5,
            (self.min_point.z + self.max_point.z) * 0.5,
        )

    def size(self) -> Vec3:
        """Retourne les dimensions de l'AABB.

        Returns:
            Vecteur (largeur, hauteur, profondeur).
        """
        return Vec3(
            self.max_point.x - self.min_point.x,
            self.max_point.y - self.min_point.y,
            self.max_point.z - self.min_point.z,
        )

    def __repr__(self) -> str:
        return f"AABB(min={self.min_point}, max={self.max_point})"


class Ray:
    """Rayon 3D défini par une origine et une direction."""

    __slots__ = ('origin', 'direction')

    def __init__(self, origin: Vec3, direction: Vec3):
        """Initialise un rayon.

        Args:
            origin: Point d'origine du rayon.
            direction: Direction du rayon (sera normalisée).
        """
        self.origin = origin
        self.direction = direction.normalized()

    def point_at(self, t: float) -> Vec3:
        """Retourne le point sur le rayon à la distance t.

        Args:
            t: Distance le long du rayon.

        Returns:
            Point à la position origin + direction * t.
        """
        return self.origin + self.direction * t

    def __repr__(self) -> str:
        return f"Ray(origin={self.origin}, dir={self.direction})"


def ray_aabb_intersect(ray: Ray, aabb: AABB) -> float | None:
    """Teste l'intersection rayon-AABB par la méthode des slabs.

    Args:
        ray: Le rayon à tester.
        aabb: La boîte englobante.

    Returns:
        Distance t du point d'intersection, ou None si pas d'intersection.
    """
    dir_data = ray.direction._data
    orig_data = ray.origin._data
    min_data = aabb.min_point._data
    max_data = aabb.max_point._data

    t_min = -np.inf
    t_max = np.inf

    for i in range(3):
        if abs(dir_data[i]) < 1e-8:
            if orig_data[i] < min_data[i] or orig_data[i] > max_data[i]:
                return None
        else:
            inv_d = 1.0 / dir_data[i]
            t1 = (min_data[i] - orig_data[i]) * inv_d
            t2 = (max_data[i] - orig_data[i]) * inv_d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return None

    if t_max < 0.0:
        return None

    return float(t_min) if t_min >= 0.0 else float(t_max)
