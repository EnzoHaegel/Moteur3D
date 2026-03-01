import numpy as np
from ..math3d import Vec3
from .material import PhysicsMaterial


POSITION_CORRECTION_PERCENT = 0.4
POSITION_SLOP = 0.01


class Contact:
    """Information de contact entre deux objets en collision."""

    __slots__ = ('obj_a', 'obj_b', 'normal', 'penetration', 'point')

    def __init__(self, obj_a, obj_b, normal: Vec3, penetration: float, point: Vec3):
        """Initialise un contact.

        Args:
            obj_a: Premier objet de scène.
            obj_b: Deuxième objet de scène.
            normal: Normale de contact (de A vers B).
            penetration: Profondeur de pénétration.
            point: Point de contact approximatif.
        """
        self.obj_a = obj_a
        self.obj_b = obj_b
        self.normal = normal
        self.penetration = penetration
        self.point = point


def detect_contact(obj_a, obj_b) -> Contact | None:
    """Détecte un contact AABB entre deux objets de scène.

    Args:
        obj_a: Premier objet.
        obj_b: Deuxième objet.

    Returns:
        Contact si collision, None sinon.
    """
    aabb_a = obj_a.get_aabb()
    aabb_b = obj_b.get_aabb()

    if not aabb_a.intersects(aabb_b):
        return None

    overlap_x = min(aabb_a.max_point.x, aabb_b.max_point.x) - \
        max(aabb_a.min_point.x, aabb_b.min_point.x)
    overlap_y = min(aabb_a.max_point.y, aabb_b.max_point.y) - \
        max(aabb_a.min_point.y, aabb_b.min_point.y)
    overlap_z = min(aabb_a.max_point.z, aabb_b.max_point.z) - \
        max(aabb_a.min_point.z, aabb_b.min_point.z)

    center_a = aabb_a.center()
    center_b = aabb_b.center()

    if overlap_x <= overlap_y and overlap_x <= overlap_z:
        sign = 1.0 if center_a.x < center_b.x else -1.0
        normal = Vec3(sign, 0.0, 0.0)
        penetration = overlap_x
    elif overlap_y <= overlap_z:
        sign = 1.0 if center_a.y < center_b.y else -1.0
        normal = Vec3(0.0, sign, 0.0)
        penetration = overlap_y
    else:
        sign = 1.0 if center_a.z < center_b.z else -1.0
        normal = Vec3(0.0, 0.0, sign)
        penetration = overlap_z

    point = Vec3(
        (max(aabb_a.min_point.x, aabb_b.min_point.x) +
         min(aabb_a.max_point.x, aabb_b.max_point.x)) * 0.5,
        (max(aabb_a.min_point.y, aabb_b.min_point.y) +
         min(aabb_a.max_point.y, aabb_b.max_point.y)) * 0.5,
        (max(aabb_a.min_point.z, aabb_b.min_point.z) +
         min(aabb_a.max_point.z, aabb_b.max_point.z)) * 0.5,
    )

    return Contact(obj_a, obj_b, normal, penetration, point)


def resolve_collision(contact: Contact):
    """Résout une collision par impulsion avec friction.

    Args:
        contact: Information de contact à résoudre.
    """
    rb_a = contact.obj_a.rigidbody
    rb_b = contact.obj_b.rigidbody

    if rb_a is None or rb_b is None:
        return
    if rb_a.is_static and rb_b.is_static:
        return

    inv_mass_sum = rb_a.inv_mass + rb_b.inv_mass
    if inv_mass_sum == 0.0:
        return

    rel_vel = rb_b.velocity - rb_a.velocity
    vel_along_normal = rel_vel.dot(contact.normal)

    if vel_along_normal > 0.0:
        _correct_position(contact, inv_mass_sum)
        return

    mat_a = rb_a.material
    mat_b = rb_b.material
    e = PhysicsMaterial.combine_restitution(mat_a, mat_b)

    j = -(1.0 + e) * vel_along_normal / inv_mass_sum

    impulse = contact.normal * j
    rb_a.add_impulse(impulse * -1.0)
    rb_b.add_impulse(impulse)

    _apply_friction(contact, j, inv_mass_sum)
    _correct_position(contact, inv_mass_sum)


def _apply_friction(contact: Contact, normal_impulse: float, inv_mass_sum: float):
    """Applique l'impulsion de friction tangentielle.

    Args:
        contact: Information de contact.
        normal_impulse: Magnitude de l'impulsion normale.
        inv_mass_sum: Somme des masses inverses.
    """
    rb_a = contact.obj_a.rigidbody
    rb_b = contact.obj_b.rigidbody

    rel_vel = rb_b.velocity - rb_a.velocity
    vel_along_normal = rel_vel.dot(contact.normal)
    tangent = rel_vel - contact.normal * vel_along_normal

    tangent_len = tangent.length()
    if tangent_len < 1e-8:
        return
    tangent = tangent * (1.0 / tangent_len)

    jt = -rel_vel.dot(tangent) / inv_mass_sum

    mu = PhysicsMaterial.combine_friction(rb_a.material, rb_b.material)

    if abs(jt) < abs(normal_impulse) * mu:
        friction_impulse = tangent * jt
    else:
        friction_impulse = tangent * (-normal_impulse * mu)

    rb_a.add_impulse(friction_impulse * -1.0)
    rb_b.add_impulse(friction_impulse)


def _correct_position(contact: Contact, inv_mass_sum: float):
    """Corrige la position pour éviter l'enfoncement (Baumgarte stabilization).

    Args:
        contact: Information de contact.
        inv_mass_sum: Somme des masses inverses.
    """
    correction_magnitude = max(contact.penetration - POSITION_SLOP, 0.0)
    correction_magnitude = correction_magnitude / \
        inv_mass_sum * POSITION_CORRECTION_PERCENT

    correction = contact.normal * correction_magnitude

    rb_a = contact.obj_a.rigidbody
    rb_b = contact.obj_b.rigidbody

    if not rb_a.is_static:
        contact.obj_a.transform.position = (
            contact.obj_a.transform.position - correction * rb_a.inv_mass
        )
    if not rb_b.is_static:
        contact.obj_b.transform.position = (
            contact.obj_b.transform.position + correction * rb_b.inv_mass
        )
