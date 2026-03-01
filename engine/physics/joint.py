import math
import numpy as np
from ..math3d import Vec3


def _rotate_vec_by_euler(v: Vec3, rotation: Vec3) -> Vec3:
    """Tourne un vecteur local en espace monde via les angles d'Euler (degrés, XYZ).

    Args:
        v: Vecteur en espace local.
        rotation: Rotation Euler en degrés (x, y, z).

    Returns:
        Vecteur tourné en espace monde.
    """
    rx = math.radians(rotation.x)
    ry = math.radians(rotation.y)
    rz = math.radians(rotation.z)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    x = v.x * (cy * cz) + v.y * (sx * sy * cz - cx * sz) + v.z * (cx * sy * cz + sx * sz)
    y = v.x * (cy * sz) + v.y * (sx * sy * sz + cx * cz) + v.z * (cx * sy * sz - sx * cz)
    z = v.x * (-sy) + v.y * (sx * cy) + v.z * (cx * cy)

    return Vec3(x, y, z)


BAUMGARTE_BIAS = 0.2
VELOCITY_DAMPING = 0.98


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
        """Retourne le point d'ancrage A en espace monde (avec rotation)."""
        rotated = _rotate_vec_by_euler(self.anchor_a, self.obj_a.transform.rotation)
        return self.obj_a.transform.position + rotated

    def _get_world_anchor_b(self) -> Vec3:
        """Retourne le point d'ancrage B en espace monde (avec rotation)."""
        rotated = _rotate_vec_by_euler(self.anchor_b, self.obj_b.transform.rotation)
        return self.obj_b.transform.position + rotated

    def solve(self, dt: float):
        """Résout la contrainte du joint.

        Args:
            dt: Pas de temps en secondes.
        """
        raise NotImplementedError


def _solve_position_baumgarte(joint, dt: float):
    """Résout la contrainte de position par stabilisation de Baumgarte.

    Args:
        joint: Le joint dont les ancres doivent coincider.
        dt: Pas de temps en secondes.
    """
    world_a = joint._get_world_anchor_a()
    world_b = joint._get_world_anchor_b()
    error = world_b - world_a

    error_len = error.length()
    if error_len < 1e-6:
        return

    rb_a = joint.obj_a.rigidbody
    rb_b = joint.obj_b.rigidbody
    if rb_a is None or rb_b is None:
        return

    inv_mass_sum = rb_a.inv_mass + rb_b.inv_mass
    if inv_mass_sum == 0.0:
        return

    direction = error * (1.0 / error_len)

    bias = (BAUMGARTE_BIAS / dt) * error_len

    rel_vel = rb_b.velocity - rb_a.velocity
    vel_along = rel_vel.dot(direction)

    lambda_val = -(vel_along + bias) / inv_mass_sum

    impulse = direction * lambda_val
    rb_a.add_impulse(impulse * -1.0)
    rb_b.add_impulse(impulse)


class HingeJoint(Joint):
    """Articulation pivot autour d'un axe (genou, coude, charnière)."""

    __slots__ = (
        'axis', 'min_angle', 'max_angle',
        'motor_speed', 'motor_max_force', 'motor_enabled',
        '_current_angle',
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
        """
        super().__init__(obj_a, obj_b, anchor_a, anchor_b)
        self.axis = axis if axis else Vec3(1.0, 0.0, 0.0)
        self.min_angle = math.radians(min_angle)
        self.max_angle = math.radians(max_angle)
        self.motor_speed = 0.0
        self.motor_max_force = motor_max_force
        self.motor_enabled = False
        self._current_angle = 0.0

    @property
    def current_angle(self) -> float:
        """Angle actuel du joint en degrés."""
        return math.degrees(self._current_angle)

    def solve(self, dt: float):
        """Résout la contrainte du joint charnière.

        Args:
            dt: Pas de temps en secondes.
        """
        if not self.active:
            return

        _solve_position_baumgarte(self, dt)
        self._solve_angle_limits(dt)
        self._solve_motor(dt)

    def _solve_angle_limits(self, dt: float):
        """Contraint l'angle entre les limites."""
        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_b is None or rb_b.is_static:
            return

        rot_a = self.obj_a.transform.rotation
        rot_b = self.obj_b.transform.rotation

        if abs(self.axis.x) > 0.5:
            self._current_angle = math.radians(rot_b.x - rot_a.x)
        elif abs(self.axis.y) > 0.5:
            self._current_angle = math.radians(rot_b.y - rot_a.y)
        else:
            self._current_angle = math.radians(rot_b.z - rot_a.z)

        correction = 0.0
        if self._current_angle < self.min_angle:
            correction = self.min_angle - self._current_angle
        elif self._current_angle > self.max_angle:
            correction = self.max_angle - self._current_angle

        if abs(correction) < 1e-6:
            return

        torque_mag = correction * (BAUMGARTE_BIAS / dt) * 0.5

        inv_inertia_sum = 0.0
        if rb_a is not None and not rb_a.is_static:
            inv_inertia_sum += rb_a.inv_inertia
        inv_inertia_sum += rb_b.inv_inertia
        if inv_inertia_sum == 0.0:
            return

        lambda_val = torque_mag / inv_inertia_sum
        torque = self.axis * lambda_val

        rb_b.add_angular_impulse(torque)
        if rb_a is not None and not rb_a.is_static:
            rb_a.add_angular_impulse(torque * -1.0)

    def _solve_motor(self, dt: float):
        """Applique le couple du moteur avec réaction sur les deux corps."""
        if not self.motor_enabled:
            return

        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_b is None or rb_b.is_static:
            return

        if abs(self.axis.x) > 0.5:
            current_vel_b = rb_b.angular_velocity.x
            current_vel_a = rb_a.angular_velocity.x if (rb_a and not rb_a.is_static) else 0.0
        elif abs(self.axis.y) > 0.5:
            current_vel_b = rb_b.angular_velocity.y
            current_vel_a = rb_a.angular_velocity.y if (rb_a and not rb_a.is_static) else 0.0
        else:
            current_vel_b = rb_b.angular_velocity.z
            current_vel_a = rb_a.angular_velocity.z if (rb_a and not rb_a.is_static) else 0.0

        relative_vel = current_vel_b - current_vel_a
        target_vel = math.radians(self.motor_speed)
        delta_omega = target_vel - relative_vel

        inv_inertia_sum = rb_b.inv_inertia
        if rb_a is not None and not rb_a.is_static:
            inv_inertia_sum += rb_a.inv_inertia
        if inv_inertia_sum == 0.0:
            return

        desired_impulse = delta_omega / inv_inertia_sum
        max_impulse = self.motor_max_force * dt
        impulse_mag = max(-max_impulse, min(max_impulse, desired_impulse))

        torque = self.axis * impulse_mag
        rb_b.add_angular_impulse(torque)
        if rb_a is not None and not rb_a.is_static:
            rb_a.add_angular_impulse(torque * -1.0)

    def __repr__(self) -> str:
        return (
            f"HingeJoint(axis={self.axis}, "
            f"angle={math.degrees(self._current_angle):.1f}°, "
            f"limits=[{math.degrees(self.min_angle):.0f}, {math.degrees(self.max_angle):.0f}])"
        )


class BallJoint(Joint):
    """Articulation sphérique (épaule, hanche)."""

    __slots__ = ()

    def __init__(
        self,
        obj_a,
        obj_b,
        anchor_a: Vec3 = None,
        anchor_b: Vec3 = None,
    ):
        """Initialise un joint sphérique.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            anchor_a: Point d'ancrage local sur obj_a.
            anchor_b: Point d'ancrage local sur obj_b.
        """
        super().__init__(obj_a, obj_b, anchor_a, anchor_b)

    def solve(self, dt: float):
        """Résout la contrainte du joint sphérique.

        Args:
            dt: Pas de temps en secondes.
        """
        if not self.active:
            return

        _solve_position_baumgarte(self, dt)

    def __repr__(self) -> str:
        return f"BallJoint()"


class FixedJoint(Joint):
    """Articulation fixe (soudure) — empêche tout mouvement relatif."""

    __slots__ = ()

    def __init__(
        self,
        obj_a,
        obj_b,
        anchor_a: Vec3 = None,
        anchor_b: Vec3 = None,
    ):
        """Initialise un joint fixe.

        Args:
            obj_a: Premier objet (parent).
            obj_b: Deuxième objet (enfant).
            anchor_a: Point d'ancrage local sur obj_a.
            anchor_b: Point d'ancrage local sur obj_b.
        """
        super().__init__(obj_a, obj_b, anchor_a, anchor_b)

    def solve(self, dt: float):
        """Résout la contrainte du joint fixe (position + rotation).

        Args:
            dt: Pas de temps en secondes.
        """
        if not self.active:
            return

        _solve_position_baumgarte(self, dt)
        self._solve_rotation(dt)

    def _solve_rotation(self, dt: float):
        """Contrainte de rotation (empêche la rotation relative)."""
        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_b is None or rb_b.is_static:
            return

        rot_diff_x = math.radians(self.obj_b.transform.rotation.x - self.obj_a.transform.rotation.x)
        rot_diff_y = math.radians(self.obj_b.transform.rotation.y - self.obj_a.transform.rotation.y)
        rot_diff_z = math.radians(self.obj_b.transform.rotation.z - self.obj_a.transform.rotation.z)

        bias_over_dt = BAUMGARTE_BIAS / dt

        inv_inertia_sum = rb_b.inv_inertia
        if rb_a is not None and not rb_a.is_static:
            inv_inertia_sum += rb_a.inv_inertia
        if inv_inertia_sum == 0.0:
            return

        for axis_vec, angle_err, vel_b_fn, vel_a_fn in [
            (Vec3(1, 0, 0), rot_diff_x,
             lambda: rb_b.angular_velocity.x,
             lambda: rb_a.angular_velocity.x if (rb_a and not rb_a.is_static) else 0.0),
            (Vec3(0, 1, 0), rot_diff_y,
             lambda: rb_b.angular_velocity.y,
             lambda: rb_a.angular_velocity.y if (rb_a and not rb_a.is_static) else 0.0),
            (Vec3(0, 0, 1), rot_diff_z,
             lambda: rb_b.angular_velocity.z,
             lambda: rb_a.angular_velocity.z if (rb_a and not rb_a.is_static) else 0.0),
        ]:
            rel_ang_vel = vel_b_fn() - vel_a_fn()
            lambda_val = -(rel_ang_vel + bias_over_dt * angle_err) / inv_inertia_sum
            impulse = axis_vec * lambda_val
            rb_b.add_angular_impulse(impulse)
            if rb_a is not None and not rb_a.is_static:
                rb_a.add_angular_impulse(impulse * -1.0)

    def __repr__(self) -> str:
        return f"FixedJoint()"
