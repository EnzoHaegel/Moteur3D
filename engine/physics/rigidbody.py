import numpy as np
from ..math3d import Vec3
from .material import PhysicsMaterial


class RigidBody:
    """Corps rigide avec masse, vélocité et accumulation de forces."""

    __slots__ = (
        'mass', '_inv_mass',
        'velocity', 'angular_velocity',
        '_force', '_torque',
        'is_kinematic', 'material',
        'gravity_scale',
    )

    def __init__(
        self,
        mass: float = 1.0,
        material: PhysicsMaterial = None,
        is_kinematic: bool = False,
    ):
        """Initialise un corps rigide.

        Args:
            mass: Masse en kg. 0 = corps statique (masse infinie).
            material: Matériau physique du corps.
            is_kinematic: Si True, non affecté par les forces mais participe aux collisions.
        """
        self.mass = mass
        self._inv_mass = 0.0 if mass <= 0.0 or is_kinematic else 1.0 / mass
        self.velocity = Vec3(0.0, 0.0, 0.0)
        self.angular_velocity = Vec3(0.0, 0.0, 0.0)
        self._force = Vec3(0.0, 0.0, 0.0)
        self._torque = Vec3(0.0, 0.0, 0.0)
        self.is_kinematic = is_kinematic
        self.material = material if material else PhysicsMaterial.DEFAULT.copy()
        self.gravity_scale = 1.0

    @property
    def inv_mass(self) -> float:
        """Inverse de la masse (0 pour les corps statiques)."""
        return self._inv_mass

    @property
    def is_static(self) -> bool:
        """True si le corps a une masse infinie (ne bouge pas)."""
        return self._inv_mass == 0.0

    def add_force(self, force: Vec3):
        """Ajoute une force au centre de masse.

        Args:
            force: Force en Newtons à appliquer.
        """
        if self.is_static:
            return
        self._force = self._force + force

    def add_torque(self, torque: Vec3):
        """Ajoute un couple (torque) au corps.

        Args:
            torque: Couple en N·m à appliquer.
        """
        if self.is_static:
            return
        self._torque = self._torque + torque

    def add_impulse(self, impulse: Vec3):
        """Applique une impulsion instantanée (modifie directement la vélocité).

        Args:
            impulse: Impulsion en kg·m/s.
        """
        if self.is_static:
            return
        self.velocity = self.velocity + impulse * self._inv_mass

    def add_angular_impulse(self, impulse: Vec3):
        """Applique une impulsion angulaire instantanée.

        Args:
            impulse: Impulsion angulaire.
        """
        if self.is_static:
            return
        self.angular_velocity = self.angular_velocity + impulse * self._inv_mass

    def integrate_forces(self, dt: float):
        """Intègre les forces accumulées en vélocité (semi-implicite Euler).

        Args:
            dt: Pas de temps en secondes.
        """
        if self.is_static:
            return
        acceleration = self._force * self._inv_mass
        self.velocity = self.velocity + acceleration * dt

        angular_accel = self._torque * self._inv_mass
        self.angular_velocity = self.angular_velocity + angular_accel * dt

    def integrate_velocity(self, position: Vec3, rotation: Vec3, dt: float) -> tuple:
        """Intègre la vélocité en position et rotation.

        Args:
            position: Position actuelle.
            rotation: Rotation actuelle en degrés (euler XYZ).
            dt: Pas de temps en secondes.

        Returns:
            Tuple (nouvelle_position, nouvelle_rotation).
        """
        if self.is_static:
            return position, rotation

        new_pos = position + self.velocity * dt

        angular_deg = self.angular_velocity * (180.0 / np.pi) * dt
        new_rot = rotation + angular_deg

        return new_pos, new_rot

    def clear_forces(self):
        """Remet les accumulateurs de forces et de couples à zéro."""
        self._force = Vec3(0.0, 0.0, 0.0)
        self._torque = Vec3(0.0, 0.0, 0.0)

    def kinetic_energy(self) -> float:
        """Calcule l'énergie cinétique du corps.

        Returns:
            Énergie cinétique en Joules.
        """
        if self.is_static:
            return 0.0
        v2 = self.velocity.length_squared()
        return 0.5 * self.mass * v2

    def __repr__(self) -> str:
        return (
            f"RigidBody(mass={self.mass:.2f}, "
            f"vel={self.velocity}, "
            f"static={self.is_static})"
        )
