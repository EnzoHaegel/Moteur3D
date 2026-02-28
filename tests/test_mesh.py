from engine.mesh import Mesh, OBJLoader
import unittest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestMesh(unittest.TestCase):
    """Tests unitaires pour la classe Mesh."""

    def _make_triangle_mesh(self) -> Mesh:
        """Crée un maillage triangulaire simple pour les tests."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return Mesh(vertices, faces, name="triangle")

    def _make_quad_mesh(self) -> Mesh:
        """Crée un maillage avec deux triangles (un quad)."""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
        ], dtype=np.int32)
        return Mesh(vertices, faces, name="quad")

    def test_vertex_count(self):
        """Nombre de sommets correct."""
        mesh = self._make_triangle_mesh()
        self.assertEqual(mesh.vertex_count(), 3)

    def test_face_count(self):
        """Nombre de faces correct."""
        mesh = self._make_triangle_mesh()
        self.assertEqual(mesh.face_count(), 1)

    def test_quad_face_count(self):
        """Quad a deux faces."""
        mesh = self._make_quad_mesh()
        self.assertEqual(mesh.face_count(), 2)

    def test_normals_computed(self):
        """Les normales sont calculées après construction."""
        mesh = self._make_triangle_mesh()
        self.assertIsNotNone(mesh.normals)
        self.assertIsNotNone(mesh.face_normals)

    def test_face_normal_direction(self):
        """La normale d'un triangle dans le plan XY pointe vers Z."""
        mesh = self._make_triangle_mesh()
        normal = mesh.face_normals[0]
        self.assertAlmostEqual(abs(normal[2]), 1.0, places=4)

    def test_normals_shape(self):
        """Les normales par sommet ont la bonne forme."""
        mesh = self._make_triangle_mesh()
        self.assertEqual(mesh.normals.shape, (3, 3))

    def test_face_normals_shape(self):
        """Les normales par face ont la bonne forme."""
        mesh = self._make_triangle_mesh()
        self.assertEqual(mesh.face_normals.shape, (1, 3))

    def test_bounds(self):
        """Bornes du maillage correctes."""
        mesh = self._make_triangle_mesh()
        bmin, bmax = mesh.get_bounds()
        np.testing.assert_array_almost_equal(bmin, [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(bmax, [1.0, 1.0, 0.0])

    def test_center(self):
        """Centre géométrique correct."""
        mesh = self._make_triangle_mesh()
        center = mesh.get_center()
        self.assertAlmostEqual(center[0], 0.5, places=4)
        self.assertAlmostEqual(center[1], 0.5, places=4)
        self.assertAlmostEqual(center[2], 0.0, places=4)

    def test_name(self):
        """Le nom du maillage est stocké."""
        mesh = self._make_triangle_mesh()
        self.assertEqual(mesh.name, "triangle")


class TestOBJLoader(unittest.TestCase):
    """Tests unitaires pour le chargeur OBJ."""

    def _write_temp_obj(self, content: str) -> str:
        """Écrit un fichier OBJ temporaire et retourne son chemin."""
        fd, path = tempfile.mkstemp(suffix='.obj')
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        return path

    def test_load_simple_triangle(self):
        """Charger un triangle simple depuis un OBJ."""
        obj_content = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
"""
        path = self._write_temp_obj(obj_content)
        try:
            mesh = OBJLoader.load(path)
            self.assertEqual(mesh.vertex_count(), 3)
            self.assertEqual(mesh.face_count(), 1)
        finally:
            os.unlink(path)

    def test_load_with_normals_uvs(self):
        """Charger un OBJ avec normales et UVs (format v/vt/vn)."""
        obj_content = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
vt 0.0 0.0
vt 1.0 0.0
vt 0.0 1.0
vn 0.0 0.0 1.0
f 1/1/1 2/2/1 3/3/1
"""
        path = self._write_temp_obj(obj_content)
        try:
            mesh = OBJLoader.load(path)
            self.assertEqual(mesh.vertex_count(), 3)
            self.assertEqual(mesh.face_count(), 1)
        finally:
            os.unlink(path)

    def test_load_quad_triangulated(self):
        """Un quad dans un OBJ est triangulé en deux triangles."""
        obj_content = """
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
f 1 2 3 4
"""
        path = self._write_temp_obj(obj_content)
        try:
            mesh = OBJLoader.load(path)
            self.assertEqual(mesh.face_count(), 2)
        finally:
            os.unlink(path)

    def test_load_with_comments(self):
        """Les lignes de commentaires sont ignorées."""
        obj_content = """
# Ceci est un commentaire
v 0.0 0.0 0.0
v 1.0 0.0 0.0
# Encore un commentaire
v 0.0 1.0 0.0
f 1 2 3
"""
        path = self._write_temp_obj(obj_content)
        try:
            mesh = OBJLoader.load(path)
            self.assertEqual(mesh.vertex_count(), 3)
        finally:
            os.unlink(path)

    def test_load_empty_raises(self):
        """Un fichier OBJ vide lève une ValueError."""
        obj_content = "# Nothing here\n"
        path = self._write_temp_obj(obj_content)
        try:
            with self.assertRaises(ValueError):
                OBJLoader.load(path)
        finally:
            os.unlink(path)

    def test_load_file_not_found(self):
        """Un chemin inexistant lève une FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            OBJLoader.load("/nonexistent/path/model.obj")


if __name__ == '__main__':
    unittest.main()
