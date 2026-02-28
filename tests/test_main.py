import unittest
from unittest.mock import patch, MagicMock, call
import sys
import os

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


class TestMainModelFound(unittest.TestCase):
    """Tests pour main() quand le modèle existe."""

    @patch('main.Engine')
    @patch('main.os.path.exists', return_value=True)
    def test_main_loads_model(self, mock_exists, MockEngine):
        engine_inst = MagicMock()
        mock_mesh = MagicMock()
        mock_mesh.name = "FinalBaseMesh.obj"
        mock_mesh.vertex_count.return_value = 100
        mock_mesh.face_count.return_value = 50
        mock_mesh.get_center.return_value = [0.0, 1.0, 0.0]
        engine_inst.load_mesh.return_value = mock_mesh
        MockEngine.return_value = engine_inst

        from main import main
        main()

        engine_inst.load_mesh.assert_called_once()
        engine_inst.run.assert_called_once()

    @patch('main.Engine')
    @patch('main.os.path.exists', return_value=True)
    def test_main_prints_model_info(self, mock_exists, MockEngine):
        engine_inst = MagicMock()
        mock_mesh = MagicMock()
        mock_mesh.name = "Test.obj"
        mock_mesh.vertex_count.return_value = 42
        mock_mesh.face_count.return_value = 10
        mock_mesh.get_center.return_value = [1.0, 2.0, 3.0]
        engine_inst.load_mesh.return_value = mock_mesh
        MockEngine.return_value = engine_inst

        from main import main
        with patch('builtins.print') as mock_print:
            main()
        printed = ''.join(str(c) for c in mock_print.call_args_list)
        self.assertIn("Test.obj", printed)


class TestMainModelNotFound(unittest.TestCase):
    """Tests pour main() quand le modèle n'existe pas."""

    @patch('main.Engine')
    @patch('main.os.path.exists', return_value=False)
    def test_main_runs_without_model(self, mock_exists, MockEngine):
        engine_inst = MagicMock()
        MockEngine.return_value = engine_inst

        from main import main
        main()

        engine_inst.load_mesh.assert_not_called()
        engine_inst.run.assert_called_once()

    @patch('main.Engine')
    @patch('main.os.path.exists', return_value=False)
    def test_main_prints_not_found(self, mock_exists, MockEngine):
        engine_inst = MagicMock()
        MockEngine.return_value = engine_inst

        from main import main
        with patch('builtins.print') as mock_print:
            main()
        printed = ''.join(str(c) for c in mock_print.call_args_list)
        self.assertIn("non trouve", printed)


if __name__ == '__main__':
    unittest.main()
