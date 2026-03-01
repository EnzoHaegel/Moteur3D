from ..math3d import Vec3
from .forces import Gravity, Drag, BuoyancyZone, Spring
from .solver import detect_contact, resolve_collision


class PhysicsWorld:
    """Monde physique orchestrant la simulation pas à pas."""

    __slots__ = (
        'gravity', '_drag', '_objects',
        '_joints', '_buoyancy_zones', '_springs',
        '_fixed_dt', '_accumulator',
        '_solver_iterations',
    )

    def __init__(
        self,
        gravity: Vec3 = None,
        fixed_dt: float = 1.0 / 120.0,
        solver_iterations: int = 8,
    ):
        """Initialise le monde physique.

        Args:
            gravity: Vecteur d'accélération gravitationnelle.
            fixed_dt: Pas de temps fixe en secondes pour le solver.
            solver_iterations: Nombre d'itérations du solveur de contraintes.
        """
        self.gravity = Gravity(gravity if gravity else Vec3(0.0, -9.81, 0.0))
        self._drag = Drag()
        self._objects = []
        self._joints = []
        self._buoyancy_zones = []
        self._springs = []
        self._fixed_dt = fixed_dt
        self._accumulator = 0.0
        self._solver_iterations = solver_iterations

    @property
    def fixed_dt(self) -> float:
        """Pas de temps fixe du solver."""
        return self._fixed_dt

    @fixed_dt.setter
    def fixed_dt(self, value: float):
        self._fixed_dt = max(1e-6, value)

    @property
    def joints(self) -> list:
        """Liste des joints actifs."""
        return self._joints

    @property
    def buoyancy_zones(self) -> list:
        """Liste des zones de flottabilité."""
        return self._buoyancy_zones

    @property
    def springs(self) -> list:
        """Liste des ressorts."""
        return self._springs

    def register(self, obj):
        """Enregistre un objet de scène dans le monde physique.

        Args:
            obj: SceneObject avec un rigidbody.
        """
        if obj not in self._objects:
            self._objects.append(obj)

    def unregister(self, obj):
        """Retire un objet du monde physique.

        Args:
            obj: SceneObject à retirer.
        """
        if obj in self._objects:
            self._objects.remove(obj)

    def add_hinge_joint(self, obj_a, obj_b, **kwargs):
        """Crée et ajoute un joint charnière.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            **kwargs: Arguments du HingeJoint (anchor_a, anchor_b, axis, etc.).

        Returns:
            Le HingeJoint créé.
        """
        from .joint import HingeJoint
        joint = HingeJoint(obj_a, obj_b, **kwargs)
        self._joints.append(joint)
        return joint

    def add_ball_joint(self, obj_a, obj_b, **kwargs):
        """Crée et ajoute un joint sphérique.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            **kwargs: Arguments du BallJoint.

        Returns:
            Le BallJoint créé.
        """
        from .joint import BallJoint
        joint = BallJoint(obj_a, obj_b, **kwargs)
        self._joints.append(joint)
        return joint

    def add_fixed_joint(self, obj_a, obj_b, **kwargs):
        """Crée et ajoute un joint fixe.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            **kwargs: Arguments du FixedJoint.

        Returns:
            Le FixedJoint créé.
        """
        from .joint import FixedJoint
        joint = FixedJoint(obj_a, obj_b, **kwargs)
        self._joints.append(joint)
        return joint

    def remove_joint(self, joint):
        """Retire un joint du monde physique.

        Args:
            joint: Joint à retirer.
        """
        if joint in self._joints:
            self._joints.remove(joint)

    def add_buoyancy_zone(self, zone: BuoyancyZone):
        """Ajoute une zone de flottabilité.

        Args:
            zone: Zone de flottabilité à ajouter.
        """
        self._buoyancy_zones.append(zone)

    def add_spring(self, spring: Spring):
        """Ajoute un ressort.

        Args:
            spring: Ressort à ajouter.
        """
        self._springs.append(spring)

    def step(self, dt: float):
        """Avance la simulation physique d'un pas de temps.

        Args:
            dt: Temps écoulé en secondes (sera subdivisé en pas fixes).
        """
        self._accumulator += dt
        max_steps = 10
        steps = 0

        while self._accumulator >= self._fixed_dt and steps < max_steps:
            self._step_fixed(self._fixed_dt)
            self._accumulator -= self._fixed_dt
            steps += 1

    def _step_fixed(self, dt: float):
        """Exécute un pas de temps fixe de la simulation.

        Args:
            dt: Pas de temps fixe.
        """
        self._apply_forces(dt)
        self._integrate_forces(dt)
        contacts = self._detect_collisions()
        self._resolve_collisions(contacts)
        self._solve_joints(dt)
        self._integrate_velocities(dt)
        self._clear_forces()

    def _apply_forces(self, dt: float):
        """Applique toutes les forces externes aux corps rigides."""
        for obj in self._objects:
            if obj.rigidbody is None or not obj.active:
                continue
            rb = obj.rigidbody
            self.gravity.apply(rb)
            self._drag.apply(rb)

        for zone in self._buoyancy_zones:
            for obj in self._objects:
                if obj.rigidbody is None or not obj.active:
                    continue
                zone.apply(obj.rigidbody, obj.get_aabb())

        for spring in self._springs:
            spring.apply()

    def _integrate_forces(self, dt: float):
        """Intègre les forces en vélocités."""
        for obj in self._objects:
            if obj.rigidbody is None or not obj.active:
                continue
            obj.rigidbody.integrate_forces(dt)

    def _detect_collisions(self) -> list:
        """Détecte les collisions entre tous les objets.

        Returns:
            Liste des contacts détectés.
        """
        contacts = []
        active = [o for o in self._objects if o.active and o.rigidbody is not None]
        n = len(active)

        for i in range(n):
            for j in range(i + 1, n):
                obj_a = active[i]
                obj_b = active[j]

                if obj_a.rigidbody.is_static and obj_b.rigidbody.is_static:
                    continue

                contact = detect_contact(obj_a, obj_b)
                if contact is not None:
                    contacts.append(contact)

        return contacts

    def _resolve_collisions(self, contacts: list):
        """Résout les collisions détectées par impulsions itératives."""
        for _ in range(self._solver_iterations):
            for contact in contacts:
                resolve_collision(contact)

    def _solve_joints(self, dt: float):
        """Résout les contraintes de joints."""
        for _ in range(self._solver_iterations):
            for joint in self._joints:
                if joint.active:
                    joint.solve(dt)

    def _integrate_velocities(self, dt: float):
        """Intègre les vélocités en positions/rotations."""
        for obj in self._objects:
            if obj.rigidbody is None or not obj.active:
                continue
            new_pos, new_rot = obj.rigidbody.integrate_velocity(
                obj.transform.position,
                obj.transform.rotation,
                dt,
            )
            obj.transform.position = new_pos
            obj.transform.rotation = new_rot

    def _clear_forces(self):
        """Remet à zéro les accumulateurs de forces."""
        for obj in self._objects:
            if obj.rigidbody is not None:
                obj.rigidbody.clear_forces()

    def reset(self):
        """Réinitialise le monde physique."""
        self._objects.clear()
        self._joints.clear()
        self._buoyancy_zones.clear()
        self._springs.clear()
        self._accumulator = 0.0
