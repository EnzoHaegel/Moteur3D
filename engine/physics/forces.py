from ..math3d import Vec3
from ..collision import AABB


class Gravity:
    """Générateur de force gravitationnelle constante."""

    __slots__ = ('acceleration',)

    def __init__(self, acceleration: Vec3 = None):
        """Initialise la gravité.

        Args:
            acceleration: Vecteur d'accélération gravitationnelle. Par défaut 9.81 m/s² vers le bas.
        """
        self.acceleration = acceleration if acceleration else Vec3(
            0.0, -9.81, 0.0)

    def apply(self, rigidbody):
        """Applique la gravité à un corps rigide.

        Args:
            rigidbody: Corps rigide cible.
        """
        if rigidbody.is_static:
            return
        force = self.acceleration * (rigidbody.mass * rigidbody.gravity_scale)
        rigidbody.add_force(force)


class Drag:
    """Générateur de résistance de l'air / fluide."""

    def apply(self, rigidbody):
        """Applique la traînée au corps rigide selon son matériau.

        Args:
            rigidbody: Corps rigide cible.
        """
        if rigidbody.is_static:
            return

        speed = rigidbody.velocity.length()
        if speed > 1e-6:
            drag_coeff = rigidbody.material.drag
            drag_force = rigidbody.velocity * (-drag_coeff * speed)
            rigidbody.add_force(drag_force)

        ang_speed = rigidbody.angular_velocity.length()
        if ang_speed > 1e-6:
            ang_drag_coeff = rigidbody.material.angular_drag
            ang_drag = rigidbody.angular_velocity * \
                (-ang_drag_coeff * ang_speed)
            rigidbody.add_torque(ang_drag)


class BuoyancyZone:
    """Zone de flottabilité (eau, lave, etc.) définie par un volume AABB."""

    __slots__ = ('aabb', 'fluid_density', 'fluid_drag', 'surface_y')

    def __init__(
        self,
        aabb: AABB,
        fluid_density: float = 1000.0,
        fluid_drag: float = 3.0,
    ):
        """Initialise une zone de flottabilité.

        Args:
            aabb: Volume de la zone de fluide.
            fluid_density: Densité du fluide en kg/m³ (eau = 1000).
            fluid_drag: Coefficient de traînée dans le fluide.
        """
        self.aabb = aabb
        self.fluid_density = fluid_density
        self.fluid_drag = fluid_drag
        self.surface_y = aabb.max_point.y

    def apply(self, rigidbody, obj_aabb: AABB):
        """Applique la poussée d'Archimède et la traînée du fluide.

        Args:
            rigidbody: Corps rigide à affecter.
            obj_aabb: AABB du corps en espace monde.
        """
        if rigidbody.is_static:
            return

        if not self.aabb.intersects(obj_aabb):
            return

        obj_bottom = obj_aabb.min_point.y
        obj_top = obj_aabb.max_point.y
        obj_height = obj_top - obj_bottom

        if obj_height < 1e-6:
            return

        submerged_top = min(obj_top, self.surface_y)
        submerged_bottom = max(obj_bottom, self.aabb.min_point.y)

        if submerged_top <= submerged_bottom:
            return

        submerged_fraction = (submerged_top - submerged_bottom) / obj_height
        submerged_fraction = max(0.0, min(1.0, submerged_fraction))

        obj_size = obj_aabb.size()
        volume = obj_size.x * obj_size.y * obj_size.z
        submerged_volume = volume * submerged_fraction

        buoyancy_force = self.fluid_density * 9.81 * submerged_volume
        rigidbody.add_force(Vec3(0.0, buoyancy_force, 0.0))

        speed = rigidbody.velocity.length()
        if speed > 1e-6:
            drag_force = rigidbody.velocity * \
                (-self.fluid_drag * submerged_fraction * speed)
            rigidbody.add_force(drag_force)

        ang_speed = rigidbody.angular_velocity.length()
        if ang_speed > 1e-6:
            ang_drag = rigidbody.angular_velocity * \
                (-self.fluid_drag * submerged_fraction * ang_speed * 0.5)
            rigidbody.add_torque(ang_drag)


class Spring:
    """Ressort reliant deux objets de scène."""

    __slots__ = ('obj_a', 'obj_b', 'rest_length', 'stiffness', 'damping')

    def __init__(self, obj_a, obj_b, rest_length: float = None,
                 stiffness: float = 50.0, damping: float = 5.0):
        """Initialise un ressort entre deux objets.

        Args:
            obj_a: Premier objet de scène.
            obj_b: Deuxième objet de scène.
            rest_length: Longueur au repos. Si None, calculée depuis les positions actuelles.
            stiffness: Raideur du ressort en N/m.
            damping: Amortissement en N·s/m.
        """
        self.obj_a = obj_a
        self.obj_b = obj_b
        if rest_length is None:
            diff = obj_b.transform.position - obj_a.transform.position
            self.rest_length = diff.length()
        else:
            self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping

    def apply(self):
        """Applique les forces du ressort aux deux corps."""
        rb_a = self.obj_a.rigidbody
        rb_b = self.obj_b.rigidbody
        if rb_a is None or rb_b is None:
            return

        pos_a = self.obj_a.transform.position
        pos_b = self.obj_b.transform.position
        diff = pos_b - pos_a
        dist = diff.length()

        if dist < 1e-8:
            return

        direction = diff * (1.0 / dist)

        stretch = dist - self.rest_length
        spring_force = self.stiffness * stretch

        rel_vel = rb_b.velocity - rb_a.velocity
        damping_force = self.damping * rel_vel.dot(direction)

        force_magnitude = spring_force + damping_force
        force = direction * force_magnitude

        rb_a.add_force(force)
        rb_b.add_force(force * -1.0)
