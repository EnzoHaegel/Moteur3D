import math
import numpy as np
from ..math3d import Vec3


class Joint:
    """Classe de base pour les contraintes articulaires entre deux objets."""

    __slots__ = ('obj_a', 'obj_b', 'anchor_a', 'anchor_b', 'active')

    def __init__(self, obj_a, obj_b, anchor_a: Vec3 = None, anchor_b: Vec3 = None):
        """Initialise un joint entre deux objets de scène.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            anchor_a: Point d'ancrage local sur obj_a.
            anchor_b: Point d'ancrage local sur obj_b.
        """
        self.obj_a = obj_a
        self.obj_b = obj_b
        self.anchor_a = anchor_a if anchor_a else Vec3(0.0, 0.0, 0.0)
        self.anchor_b = anchor_b if anchor_b else Vec3(0.0, 0.0, 0.0)
        self.active = True

    def _get_world_anchor_a(self) -> Vec3:
        """Retourne le point d'ancrage A en espace monde."""
        return self.obj_a.transform.position + self.anchor_a

    def _get_world_anchor_b(self) -> Vec3:
        """Retourne le point d'ancrage B en espace monde."""
        return self.obj_b.transform.position + self.anchor_b

    def solve(self, dt: float):
        """Résout la contrainte du joint.

        Args:
            dt: Pas de temps en secondes.
        """
        raise NotImplementedError


class HingeJoint(Joint):
    """Articulation pivot autour d'un axe (genou, coude, charnière)."""

    __slots__ = (
        'obj_a', 'obj_b', 'anchor_a', 'anchor_b', 'active',
        'axis', 'min_angle', 'max_angle',
        'motor_speed', 'motor_max_force', 'motor_enabled',
        '_current_angle', 'stiffness', 'damping',
    )

    def __init__(
        self,
        obj_a,
        obj_b,
        anchor_a: Vec3 = None,
        anchor_b: Vec3 = None,
        axis: Vec3 = None,
        min_angle: float = -180.0,
        max_angle: float = 180.0,
        motor_max_force: float = 100.0,
        stiffness: float = 500.0,
        damping: float = 50.0,
    ):
        """Initialise un joint charnière.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            anchor_a: Point d'ancrage local sur obj_a.
            anchor_b: Point d'ancrage local sur obj_b.
            axis: Axe de rotation (espace local de obj_a).
            min_angle: Angle minimum en degrés.
            max_angle: Angle maximum en degrés.
            motor_max_force: Force maximale du moteur en N·m.
            stiffness: Raideur de la contrainte de position.
            damping: Amortissement de la contrainte de position.
        """
        super().__init__(obj_a, obj_b, anchor_a, anchor_b)
        self.axis = axis if axis else Vec3(1.0, 0.0, 0.0)
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.motor_speed = 0.0
        self.motor_max_force = motor_max_force
        self.motor_enabled = False
        self._current_angle = 0.0
        self.stiffness = stiffness
        self.damping = damping

    @property
    def current_angle(self) -> float:
        """Angle actuel du joint en degrés."""
        return self._current_angle

    def solve(self, dt: float):
        """Résout la contrainte du joint charnière.

        Args:
            dt: Pas de temps en secondes.
        """
        if not self.active:
            return

        self._solve_position_constraint(dt)
        self._solve_angle_limits(dt)
        self._solve_motor(dt)

    def _solve_position_constraint(self, dt: float):
        """Maintient les points d'ancrage ensemble."""
        world_a = self._get_world_anchor_a()
        world_b = self._get_world_anchor_b()
        error = world_b - world_a

        error_len = error.length()
        if error_len < 1e-6:
            return

        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_a is None or rb_b is None:
            return

        inv_mass_sum = rb_a.inv_mass + rb_b.inv_mass
        if inv_mass_sum == 0.0:
            return

        rel_vel = rb_b.velocity - rb_a.velocity
        direction = error * (1.0 / error_len)

        spring_force = self.stiffness * error_len
        damping_force = self.damping * rel_vel.dot(direction)
        force_mag = spring_force + damping_force

        force_mag = max(-self.stiffness * 10.0,
                        min(self.stiffness * 10.0, force_mag))
        force = direction * (force_mag * dt)

        rb_a.add_impulse(force)
        rb_b.add_impulse(force * -1.0)

    def _solve_angle_limits(self, dt: float):
        """Contraint l'angle entre les limites."""
        rb_b = self.obj_b.rigidbody
        if rb_b is None or rb_b.is_static:
            return

        rot_a = self.obj_a.transform.rotation
        rot_b = self.obj_b.transform.rotation

        if abs(self.axis.x) > 0.5:
            self._current_angle = rot_b.x - rot_a.x
        elif abs(self.axis.y) > 0.5:
            self._current_angle = rot_b.y - rot_a.y
        else:
            self._current_angle = rot_b.z - rot_a.z

        if self._current_angle < self.min_angle:
            correction = self.min_angle - self._current_angle
            torque = self.axis * (correction * self.stiffness * 0.1 * dt)
            rb_b.add_angular_impulse(torque)
        elif self._current_angle > self.max_angle:
            correction = self.max_angle - self._current_angle
            torque = self.axis * (correction * self.stiffness * 0.1 * dt)
            rb_b.add_angular_impulse(torque)

    def _solve_motor(self, dt: float):
        """Applique le couple du moteur."""
        if not self.motor_enabled:
            return

        rb_b = self.obj_b.rigidbody
        if rb_b is None or rb_b.is_static:
            return

        if abs(self.axis.x) > 0.5:
            current_vel = rb_b.angular_velocity.x
        elif abs(self.axis.y) > 0.5:
            current_vel = rb_b.angular_velocity.y
        else:
            current_vel = rb_b.angular_velocity.z

        target_vel = math.radians(self.motor_speed)
        error = target_vel - current_vel
        torque_mag = error * self.motor_max_force * dt
        torque_mag = max(-self.motor_max_force * dt,
                         min(self.motor_max_force * dt, torque_mag))

        torque = self.axis * torque_mag
        rb_b.add_angular_impulse(torque)

    def __repr__(self) -> str:
        return (
            f"HingeJoint(axis={self.axis}, "
            f"angle={self._current_angle:.1f}°, "
            f"limits=[{self.min_angle}, {self.max_angle}])"
        )


class BallJoint(Joint):
    """Articulation sphérique (épaule, hanche)."""

    __slots__ = (
        'obj_a', 'obj_b', 'anchor_a', 'anchor_b', 'active',
        'stiffness', 'damping',
    )

    def __init__(
        self,
        obj_a,
        obj_b,
        anchor_a: Vec3 = None,
        anchor_b: Vec3 = None,
        stiffness: float = 500.0,
        damping: float = 50.0,
    ):
        """Initialise un joint sphérique.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            anchor_a: Point d'ancrage local sur obj_a.
            anchor_b: Point d'ancrage local sur obj_b.
            stiffness: Raideur de la contrainte de position.
            damping: Amortissement de la contrainte de position.
        """
        super().__init__(obj_a, obj_b, anchor_a, anchor_b)
        self.stiffness = stiffness
        self.damping = damping

    def solve(self, dt: float):
        """Résout la contrainte du joint sphérique.

        Args:
            dt: Pas de temps en secondes.
        """
        if not self.active:
            return

        world_a = self._get_world_anchor_a()
        world_b = self._get_world_anchor_b()
        error = world_b - world_a

        error_len = error.length()
        if error_len < 1e-6:
            return

        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_a is None or rb_b is None:
            return

        inv_mass_sum = rb_a.inv_mass + rb_b.inv_mass
        if inv_mass_sum == 0.0:
            return

        direction = error * (1.0 / error_len)
        rel_vel = rb_b.velocity - rb_a.velocity

        spring_force = self.stiffness * error_len
        damping_force = self.damping * rel_vel.dot(direction)
        force_mag = (spring_force + damping_force) * dt

        force = direction * force_mag

        rb_a.add_impulse(force)
        rb_b.add_impulse(force * -1.0)

    def __repr__(self) -> str:
        return f"BallJoint(stiffness={self.stiffness})"


class FixedJoint(Joint):
    """Articulation fixe (soudure) — empêche tout mouvement relatif."""

    __slots__ = (
        'obj_a', 'obj_b', 'anchor_a', 'anchor_b', 'active',
        'stiffness', 'damping',
    )

    def __init__(
        self,
        obj_a,
        obj_b,
        anchor_a: Vec3 = None,
        anchor_b: Vec3 = None,
        stiffness: float = 1000.0,
        damping: float = 100.0,
    ):
        """Initialise un joint fixe.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            anchor_a: Point d'ancrage local sur obj_a.
            anchor_b: Point d'ancrage local sur obj_b.
            stiffness: Raideur de la contrainte.
            damping: Amortissement de la contrainte.
        """
        super().__init__(obj_a, obj_b, anchor_a, anchor_b)
        self.stiffness = stiffness
        self.damping = damping

    def solve(self, dt: float):
        """Résout la contrainte du joint fixe (position + rotation).

        Args:
            dt: Pas de temps en secondes.
        """
        if not self.active:
            return

        self._solve_position(dt)
        self._solve_rotation(dt)

    def _solve_position(self, dt: float):
        """Contrainte de position."""
        world_a = self._get_world_anchor_a()
        world_b = self._get_world_anchor_b()
        error = world_b - world_a

        error_len = error.length()
        if error_len < 1e-6:
            return

        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_a is None or rb_b is None:
            return

        inv_mass_sum = rb_a.inv_mass + rb_b.inv_mass
        if inv_mass_sum == 0.0:
            return

        direction = error * (1.0 / error_len)
        rel_vel = rb_b.velocity - rb_a.velocity

        spring_force = self.stiffness * error_len
        damping_force = self.damping * rel_vel.dot(direction)
        force_mag = (spring_force + damping_force) * dt

        force = direction * force_mag
        rb_a.add_impulse(force)
        rb_b.add_impulse(force * -1.0)

    def _solve_rotation(self, dt: float):
        """Contrainte de rotation (empêche la rotation relative)."""
        rb_b = self.obj_b.rigidbody
        if rb_b is None or rb_b.is_static:
            return

        rot_diff = self.obj_b.transform.rotation - self.obj_a.transform.rotation
        correction = rot_diff * (-self.stiffness * 0.01 * dt)

        ang_vel_diff = Vec3(0.0, 0.0, 0.0)
        rb_a = self.obj_a.rigidbody
        if rb_a is not None:
            ang_vel_diff = rb_b.angular_velocity - rb_a.angular_velocity
        else:
            ang_vel_diff = rb_b.angular_velocity

        damping_correction = ang_vel_diff * (-self.damping * 0.01 * dt)
        total = correction + damping_correction

        rb_b.add_angular_impulse(total)

    def __repr__(self) -> str:
        return f"FixedJoint(stiffness={self.stiffness})"
