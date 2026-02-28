from engine.engine import Engine
from engine.math3d import Mat4
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def main():
    """Point d'entrée principal du moteur 3D."""
    engine = Engine(
        width=1280,
        height=720,
        title="Moteur 3D - ZQSD + Souris",
    )

    model_path = os.path.join(os.path.dirname(
        __file__), "model", "FinalBaseMesh.obj")

    if os.path.exists(model_path):
        model_matrix = Mat4.scale(1.0, 1.0, 1.0)
        mesh = engine.load_mesh(model_path, model_matrix)
        center = mesh.get_center()
        print(f"Modele charge: {mesh.name}")
        print(f"  Sommets: {mesh.vertex_count()}")
        print(f"  Triangles: {mesh.face_count()}")
        print(f"  Centre: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    else:
        print(f"Fichier modele non trouve: {model_path}")
        print("Lancement avec scene vide (grille uniquement)")

    print("\nControles:")
    print("  ZQSD / WASD : Deplacer la camera")
    print("  Souris      : Regarder autour")
    print("  Espace      : Monter")
    print("  Shift       : Descendre")
    print("  F1          : Basculer solide/fil de fer")
    print("  F2          : Afficher/Masquer la grille")
    print("  F3          : Afficher/Masquer le HUD")
    print("  Echap       : Liberer/Quitter")

    engine.run()


if __name__ == "__main__":  # pragma: no cover
    main()
