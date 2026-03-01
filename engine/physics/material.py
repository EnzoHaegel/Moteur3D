import math


class PhysicsMaterial:
    """Matériau physique définissant le comportement d'un corps rigide."""

    __slots__ = ('friction', 'restitution', 'density',
                 'drag', 'angular_drag', 'name')

    def __init__(
        self,
        friction: float = 0.5,
        restitution: float = 0.3,
        density: float = 1000.0,
        drag: float = 0.01,
        angular_drag: float = 0.05,
        name: str = "custom",
    ):
        """Initialise un matériau physique.

        Args:
            friction: Coefficient de friction (0.0 = glissant, 1.0 = rugueux).
            restitution: Coefficient de rebond (0.0 = pas de rebond, 1.0 = parfait).
            density: Densité en kg/m³.
            drag: Résistance linéaire de l'air/fluide.
            angular_drag: Résistance angulaire de l'air/fluide.
            name: Nom du matériau.
        """
        self.friction = max(0.0, min(1.0, friction))
        self.restitution = max(0.0, min(1.0, restitution))
        self.density = max(0.01, density)
        self.drag = max(0.0, drag)
        self.angular_drag = max(0.0, angular_drag)
        self.name = name

    def copy(self) -> 'PhysicsMaterial':
        """Retourne une copie du matériau.

        Returns:
            Nouveau PhysicsMaterial avec les mêmes valeurs.
        """
        return PhysicsMaterial(
            friction=self.friction,
            restitution=self.restitution,
            density=self.density,
            drag=self.drag,
            angular_drag=self.angular_drag,
            name=self.name,
        )

    def __repr__(self) -> str:
        return (
            f"PhysicsMaterial('{self.name}', "
            f"friction={self.friction:.2f}, "
            f"restitution={self.restitution:.2f}, "
            f"density={self.density:.0f})"
        )

    @staticmethod
    def combine_friction(a: 'PhysicsMaterial', b: 'PhysicsMaterial') -> float:
        """Calcule le coefficient de friction combiné entre deux matériaux.

        Args:
            a: Premier matériau.
            b: Deuxième matériau.

        Returns:
            Friction combinée (moyenne géométrique).
        """
        return math.sqrt(a.friction * b.friction)

    @staticmethod
    def combine_restitution(a: 'PhysicsMaterial', b: 'PhysicsMaterial') -> float:
        """Calcule le coefficient de rebond combiné entre deux matériaux.

        Args:
            a: Premier matériau.
            b: Deuxième matériau.

        Returns:
            Restitution combinée (maximum des deux).
        """
        return max(a.restitution, b.restitution)


PhysicsMaterial.STONE = PhysicsMaterial(
    friction=0.6, restitution=0.1, density=2500.0,
    drag=0.01, angular_drag=0.05, name="stone",
)

PhysicsMaterial.RUBBER = PhysicsMaterial(
    friction=0.9, restitution=0.8, density=1100.0,
    drag=0.02, angular_drag=0.05, name="rubber",
)

PhysicsMaterial.ICE = PhysicsMaterial(
    friction=0.05, restitution=0.1, density=917.0,
    drag=0.01, angular_drag=0.02, name="ice",
)

PhysicsMaterial.WOOD = PhysicsMaterial(
    friction=0.4, restitution=0.3, density=600.0,
    drag=0.02, angular_drag=0.05, name="wood",
)

PhysicsMaterial.METAL = PhysicsMaterial(
    friction=0.3, restitution=0.2, density=7800.0,
    drag=0.005, angular_drag=0.02, name="metal",
)

PhysicsMaterial.DEFAULT = PhysicsMaterial(
    friction=0.5, restitution=0.3, density=1000.0,
    drag=0.01, angular_drag=0.05, name="default",
)
