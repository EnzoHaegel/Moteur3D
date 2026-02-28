import pygame
import sys
import time
from .math3d import Vec3, Mat4
from .camera import Camera
from .mesh import Mesh, OBJLoader
from .renderer import Renderer


class Engine:
    """Moteur 3D principal gérant la boucle de jeu, les entrées et le rendu."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        title: str = "Moteur 3D",
    ):
        """Initialise le moteur 3D avec Pygame.

        Args:
            width: Largeur de la fenêtre.
            height: Hauteur de la fenêtre.
            title: Titre de la fenêtre.
        """
        pygame.init()
        pygame.display.set_caption(title)

        self._width = width
        self._height = height

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK,
            pygame.GL_CONTEXT_PROFILE_CORE,
        )
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_SWAP_CONTROL, 1)

        self._screen = pygame.display.set_mode(
            (width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        self._clock = pygame.time.Clock()
        self._running = False

        self._camera = Camera(
            position=Vec3(0.0, 5.0, 15.0),
            aspect=width / height,
        )
        self._renderer = Renderer(width, height)

        self._meshes: list[Mesh] = []
        self._model_matrices: list[Mat4] = []

        self._mouse_captured = True
        self._render_mode = 'solid'
        self._show_grid = True
        self._show_hud = True
        self._hud_font = None

        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    @property
    def camera(self) -> Camera:
        """Accès à la caméra du moteur."""
        return self._camera

    @property
    def renderer(self) -> Renderer:
        """Accès au renderer du moteur."""
        return self._renderer

    def load_mesh(self, filepath: str, model_matrix: Mat4 = None) -> Mesh:
        """Charge un fichier OBJ et l'ajoute à la scène.

        Args:
            filepath: Chemin vers le fichier .obj.
            model_matrix: Matrice de transformation du modèle (optionnel).

        Returns:
            Le Mesh chargé.
        """
        mesh = OBJLoader.load(filepath)
        self._meshes.append(mesh)
        self._model_matrices.append(
            model_matrix if model_matrix else Mat4.identity())
        return mesh

    def add_mesh(self, mesh: Mesh, model_matrix: Mat4 = None):
        """Ajoute un maillage existant à la scène.

        Args:
            mesh: Le maillage à ajouter.
            model_matrix: Matrice de transformation du modèle (optionnel).
        """
        self._meshes.append(mesh)
        self._model_matrices.append(
            model_matrix if model_matrix else Mat4.identity())

    def _handle_events(self) -> bool:
        """Traite les événements Pygame. Retourne False si on doit quitter."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self._mouse_captured:
                        self._mouse_captured = False
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                    else:
                        return False

                if event.key == pygame.K_F1:
                    self._render_mode = 'solid' if self._render_mode != 'solid' else 'wireframe'

                if event.key == pygame.K_F2:
                    self._show_grid = not self._show_grid

                if event.key == pygame.K_F3:
                    self._show_hud = not self._show_hud

            if event.type == pygame.MOUSEBUTTONDOWN and not self._mouse_captured:
                self._mouse_captured = True
                pygame.mouse.set_visible(False)
                pygame.event.set_grab(True)

        return True

    def _process_input(self, dt: float):
        """Traite les entrées clavier et souris."""
        if self._mouse_captured:
            dx, dy = pygame.mouse.get_rel()
            self._camera.process_mouse(float(dx), float(dy))

        pressed = pygame.key.get_pressed()
        keys = {
            'z': pressed[pygame.K_w] or pressed[pygame.K_z],
            's': pressed[pygame.K_s],
            'q': pressed[pygame.K_a] or pressed[pygame.K_q],
            'd': pressed[pygame.K_d],
            'space': pressed[pygame.K_SPACE],
            'shift': pressed[pygame.K_LSHIFT] or pressed[pygame.K_RSHIFT],
        }
        self._camera.process_keyboard(keys, dt)

    def _render(self):
        """Effectue le rendu de la scène complète."""
        self._renderer.clear()

        vp = self._camera.get_vp_matrix()

        if self._show_grid:
            self._renderer.render_grid(vp)

        for mesh, model in zip(self._meshes, self._model_matrices):
            mvp = vp @ model
            if self._render_mode == 'solid':
                self._renderer.render_mesh(mesh, mvp, model)
            else:
                self._renderer.render_wireframe(mesh, mvp)

        self._renderer.render_crosshair()

        self._render_fps()

        if self._show_hud:
            self._render_hud()

        self._renderer.present_overlay()
        pygame.display.flip()

    def _render_fps(self):
        """Affiche le compteur FPS en haut à droite."""
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont("consolas", 16)
        fps = self._clock.get_fps()
        fps_text = f"{fps:.0f} FPS"
        text_surface = self._hud_font.render(fps_text, True, (0, 255, 100))
        bg_surface = pygame.Surface(
            (text_surface.get_width() + 10, text_surface.get_height() + 6),
            pygame.SRCALPHA,
        )
        bg_surface.fill((0, 0, 0, 160))
        overlay = self._renderer.overlay
        x = self._width - text_surface.get_width() - 16
        overlay.blit(bg_surface, (x - 4, 6))
        overlay.blit(text_surface, (x, 9))

    def _render_hud(self):
        """Affiche les informations HUD (position, contrôles)."""
        if self._hud_font is None:
            self._hud_font = pygame.font.SysFont("consolas", 16)
        font = self._hud_font
        pos = self._camera.position

        lines = [
            f"Pos: ({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})",
            f"Yaw: {self._camera.yaw:.1f}  Pitch: {self._camera.pitch:.1f}",
            f"Mode: {self._render_mode} [F1]  Grille: {'ON' if self._show_grid else 'OFF'} [F2]",
            f"Meshes: {len(self._meshes)}  Triangles: {sum(m.face_count() for m in self._meshes)}",
            f"ZQSD: Bouger  Souris: Regarder  Espace/Shift: Haut/Bas  Echap: Menu",
        ]

        overlay = self._renderer.overlay
        y = 10
        for line in lines:
            text_surface = font.render(line, True, (220, 220, 220))
            bg_surface = pygame.Surface(
                (text_surface.get_width() + 8, text_surface.get_height() + 4),
                pygame.SRCALPHA,
            )
            bg_surface.fill((0, 0, 0, 140))
            overlay.blit(bg_surface, (8, y - 2))
            overlay.blit(text_surface, (12, y))
            y += 18

    def run(self):
        """Lance la boucle principale du moteur."""
        self._running = True
        last_time = time.perf_counter()

        while self._running:
            current_time = time.perf_counter()
            dt = current_time - last_time
            last_time = current_time
            dt = min(dt, 0.05)

            if not self._handle_events():
                self._running = False
                break

            self._process_input(dt)
            self._render()
            self._clock.tick()

        pygame.quit()
