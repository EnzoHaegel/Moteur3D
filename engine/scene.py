from .mesh import Mesh
from .transform import Transform
from .collision import AABB
from .math3d import Vec3


class SceneObject:
    """Objet de scène combinant un maillage, une transformation, une couleur et un corps rigide."""

    __slots__ = ('mesh', 'transform', 'color', 'name', 'active', 'rigidbody')

    def __init__(
        self,
        mesh: Mesh,
        transform: Transform = None,
        color: tuple = (0.7, 0.63, 0.55),
        name: str = "object",
        active: bool = True,
        rigidbody=None,
    ):
        """Initialise un objet de scène.

        Args:
            mesh: Le maillage de l'objet.
            transform: Transformation 3D (position, rotation, échelle).
            color: Couleur RGB normalisée (0.0 à 1.0).
            name: Nom de l'objet.
            active: Si True, l'objet est rendu et collidable.
            rigidbody: Corps rigide pour la physique (optionnel).
        """
        self.mesh = mesh
        self.transform = transform if transform else Transform()
        self.color = color
        self.name = name
        self.active = active
        self.rigidbody = rigidbody

    def get_aabb(self) -> AABB:
        """Calcule la boîte englobante en espace monde.

        Returns:
            AABB de l'objet transformé.
        """
        return AABB.from_mesh(self.mesh, self.transform)

    def __repr__(self) -> str:
        return (
            f"SceneObject(name='{self.name}', "
            f"active={self.active}, "
            f"color={self.color})"
        )
