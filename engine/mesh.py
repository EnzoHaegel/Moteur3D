import numpy as np
from typing import Tuple


class Mesh:
    """Maillage 3D stocké sous forme de tableaux NumPy optimisés."""

    __slots__ = ('vertices', 'faces', 'normals', 'face_normals', 'name')

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, name: str = "mesh"):
        """Initialise un maillage à partir de vertices (Nx3) et faces (Mx3).

        Args:
            vertices: Tableau de sommets de forme (N, 3).
            faces: Tableau d'indices de faces triangulaires de forme (M, 3).
            name: Nom du maillage.
        """
        self.vertices = vertices.astype(np.float32)
        self.faces = faces.astype(np.int32)
        self.name = name
        self.normals = None
        self.face_normals = None
        self._compute_normals()

    def _compute_normals(self):
        """Calcule les normales par face et par sommet de manière vectorisée."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0
        face_normals = np.cross(edge1, edge2)

        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        self.face_normals = (face_normals / norms).astype(np.float32)

        self.normals = np.zeros_like(self.vertices, dtype=np.float32)
        np.add.at(self.normals, self.faces[:, 0], self.face_normals)
        np.add.at(self.normals, self.faces[:, 1], self.face_normals)
        np.add.at(self.normals, self.faces[:, 2], self.face_normals)

        vert_norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        vert_norms = np.where(vert_norms < 1e-8, 1.0, vert_norms)
        self.normals = self.normals / vert_norms

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne les bornes min et max du maillage (AABB)."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def get_center(self) -> np.ndarray:
        """Retourne le centre géométrique du maillage."""
        bmin, bmax = self.get_bounds()
        return (bmin + bmax) / 2.0

    def vertex_count(self) -> int:
        """Nombre de sommets."""
        return self.vertices.shape[0]

    def face_count(self) -> int:
        """Nombre de faces triangulaires."""
        return self.faces.shape[0]


class OBJLoader:
    """Chargeur de fichiers Wavefront OBJ optimisé."""

    @staticmethod
    def load(filepath: str) -> Mesh:
        """Charge un fichier OBJ et retourne un Mesh.

        Args:
            filepath: Chemin vers le fichier .obj.

        Returns:
            Mesh contenant les vertices et faces du modèle.
        """
        vertices = []
        faces = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line[0] == '#':
                    continue

                parts = line.split()
                prefix = parts[0]

                if prefix == 'v' and len(parts) >= 4:
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])])
                elif prefix == 'f':
                    face_verts = []
                    for p in parts[1:]:
                        idx = int(p.split('/')[0]) - 1
                        face_verts.append(idx)
                    for i in range(1, len(face_verts) - 1):
                        faces.append(
                            [face_verts[0], face_verts[i], face_verts[i + 1]])

        if not vertices:
            raise ValueError(f"Aucun sommet trouvé dans {filepath}")

        verts_array = np.array(vertices, dtype=np.float32)
        faces_array = np.array(faces, dtype=np.int32) if faces else np.zeros(
            (0, 3), dtype=np.int32)

        return Mesh(verts_array, faces_array, name=filepath.split('/')[-1].split('\\')[-1])
