import numpy as np
import math
from .mesh import Mesh


class Primitives:
    """Générateur de maillages procéduraux pour les formes géométriques de base."""

    @staticmethod
    def cube(size: float = 1.0) -> Mesh:
        """Crée un cube centré à l'origine.

        Args:
            size: Longueur d'une arête du cube.

        Returns:
            Mesh du cube avec normales calculées.
        """
        h = size / 2.0
        vertices = np.array([
            [-h, -h,  h], [h, -h,  h], [h,  h,  h], [-h,  h,  h],
            [-h, -h, -h], [-h,  h, -h], [h,  h, -h], [h, -h, -h],
            [-h,  h, -h], [-h,  h,  h], [h,  h,  h], [h,  h, -h],
            [-h, -h, -h], [h, -h, -h], [h, -h,  h], [-h, -h,  h],
            [h, -h, -h], [h,  h, -h], [h,  h,  h], [h, -h,  h],
            [-h, -h, -h], [-h, -h,  h], [-h,  h,  h], [-h,  h, -h],
        ], dtype=np.float32)

        faces = np.array([
            [0,  1,  2],  [0,  2,  3],
            [4,  5,  6],  [4,  6,  7],
            [8,  9,  10], [8,  10, 11],
            [12, 13, 14], [12, 14, 15],
            [16, 17, 18], [16, 18, 19],
            [20, 21, 22], [20, 22, 23],
        ], dtype=np.int32)

        return Mesh(vertices, faces, name="cube")

    @staticmethod
    def sphere(radius: float = 1.0, segments: int = 16, rings: int = 16) -> Mesh:
        """Crée une sphère UV centrée à l'origine.

        Args:
            radius: Rayon de la sphère.
            segments: Nombre de segments horizontaux.
            rings: Nombre d'anneaux verticaux.

        Returns:
            Mesh de la sphère avec normales calculées.
        """
        vertices = []
        for i in range(rings + 1):
            phi = math.pi * i / rings
            for j in range(segments + 1):
                theta = 2.0 * math.pi * j / segments
                x = radius * math.sin(phi) * math.cos(theta)
                y = radius * math.cos(phi)
                z = radius * math.sin(phi) * math.sin(theta)
                vertices.append([x, y, z])

        faces = []
        for i in range(rings):
            for j in range(segments):
                a = i * (segments + 1) + j
                b = a + segments + 1
                faces.append([a, b, a + 1])
                faces.append([a + 1, b, b + 1])

        return Mesh(
            np.array(vertices, dtype=np.float32),
            np.array(faces, dtype=np.int32),
            name="sphere",
        )

    @staticmethod
    def cylinder(radius: float = 1.0, height: float = 2.0, segments: int = 16) -> Mesh:
        """Crée un cylindre centré à l'origine, aligné sur l'axe Y.

        Args:
            radius: Rayon du cylindre.
            height: Hauteur totale du cylindre.
            segments: Nombre de segments autour de la circonférence.

        Returns:
            Mesh du cylindre avec normales calculées.
        """
        half_h = height / 2.0
        vertices = []
        faces = []

        for i in range(segments + 1):
            theta = 2.0 * math.pi * i / segments
            x = radius * math.cos(theta)
            z = radius * math.sin(theta)
            vertices.append([x, half_h, z])
            vertices.append([x, -half_h, z])

        for i in range(segments):
            top = i * 2
            bot = top + 1
            next_top = (i + 1) * 2
            next_bot = next_top + 1
            faces.append([top, bot, next_bot])
            faces.append([top, next_bot, next_top])

        top_center = len(vertices)
        vertices.append([0.0, half_h, 0.0])
        bot_center = len(vertices)
        vertices.append([0.0, -half_h, 0.0])

        for i in range(segments):
            top = i * 2
            next_top = (i + 1) * 2
            faces.append([top_center, top, next_top])

            bot = i * 2 + 1
            next_bot = (i + 1) * 2 + 1
            faces.append([bot_center, next_bot, bot])

        return Mesh(
            np.array(vertices, dtype=np.float32),
            np.array(faces, dtype=np.int32),
            name="cylinder",
        )

    @staticmethod
    def plane(width: float = 10.0, depth: float = 10.0) -> Mesh:
        """Crée un plan horizontal centré à l'origine dans le plan XZ.

        Args:
            width: Largeur du plan (axe X).
            depth: Profondeur du plan (axe Z).

        Returns:
            Mesh du plan avec normales calculées.
        """
        hw = width / 2.0
        hd = depth / 2.0
        vertices = np.array([
            [-hw, 0.0, -hd],
            [hw, 0.0, -hd],
            [hw, 0.0,  hd],
            [-hw, 0.0,  hd],
        ], dtype=np.float32)

        faces = np.array([
            [0, 3, 2],
            [0, 2, 1],
        ], dtype=np.int32)

        return Mesh(vertices, faces, name="plane")
