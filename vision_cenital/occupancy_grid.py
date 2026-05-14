"""
occupancy_grid.py - Mapa de ocupacion en coordenadas mundo (cm).

Toma las detecciones del detector (mascara de paredes + obstaculos) y las
proyecta a una rejilla 2D en cm. El resultado sirve como entrada para
A*/BFS/Dijkstra al planificar rutas para el robot.

Sistema de coordenadas (cm), igual que homography.json:
  - Origen (0, 0) en esquina SUPERIOR IZQUIERDA del campo
  - X positivo hacia la DERECHA  (eje largo, ~404.5 cm)
  - Y positivo hacia ABAJO       (eje ancho, ~210 cm)

Cada celda del grid representa un cuadrado de cell_cm x cell_cm (default 5).
Valores posibles en cada celda:

    FREE      celda libre, el robot puede pasar
    WALL      pared fija de la pista
    OBSTACLE  obstaculo movible (cubo negro)
    INFLATED  celda libre pero dentro del radio del robot respecto a una
              pared/obstaculo. El planificador la debe tratar como bloqueada
              para que la ruta no roce los bordes.

Uso tipico:

    grid = OccupancyGrid(homography, cell_cm=5.0, robot_radius_cm=15.0)
    occ = grid.build(wall_mask, obstacles)         # array (rows, cols)
    img = grid.render(occ, color_dets=color_dets)  # imagen BGR para mostrar
"""

import cv2
import numpy as np


FREE     = 0
WALL     = 1
OBSTACLE = 2
INFLATED = 3

DEFAULT_CELL_CM         = 5.0
DEFAULT_ROBOT_RADIUS_CM = 15.0


class OccupancyGrid:
    """Construye y dibuja un grid de ocupacion en cm.

    El grid se reconstruye en cada frame: barato (un warpPerspective + un
    resize + un dilate sobre una imagen de ~400x200 px).
    """

    def __init__(self, homography, cell_cm=DEFAULT_CELL_CM,
                 robot_radius_cm=DEFAULT_ROBOT_RADIUS_CM):
        if not homography.enabled:
            raise ValueError(
                "OccupancyGrid necesita una homografia calibrada. "
                "Corre calibrate_homography.py primero.")
        self.H = homography.H
        self.field_w_cm = homography.field_w_cm
        self.field_h_cm = homography.field_h_cm
        self.cell_cm = float(cell_cm)
        self.robot_radius_cm = float(robot_radius_cm)
        self.cols = int(np.ceil(self.field_w_cm / self.cell_cm))
        self.rows = int(np.ceil(self.field_h_cm / self.cell_cm))
        # Tamano de la vista metrica intermedia (1 px = 1 cm)
        self._cm_w = int(np.ceil(self.field_w_cm))
        self._cm_h = int(np.ceil(self.field_h_cm))

    # ------------------------------------------------------------------ build
    def build(self, wall_mask, obstacles):
        """Construye el grid de ocupacion para el frame actual.

        wall_mask: imagen binaria uint8 en el espacio de pixeles del frame
                   (mismas dimensiones que el frame undistorted). Pixeles 255
                   son pared.
        obstacles: lista de detecciones de obstaculos con campo 'bbox' en
                   pixeles (x, y, w, h).

        Devuelve numpy array (rows, cols) con valores FREE/WALL/OBSTACLE/
        INFLATED.
        """
        # 1) Mascara de obstaculos por separado (la queremos diferenciar de
        #    las paredes en la visualizacion).
        obst_mask = np.zeros_like(wall_mask)
        for o in obstacles:
            x, y, w, h = o['bbox']
            cv2.rectangle(obst_mask, (x, y), (x + w, y + h), 255,
                          thickness=cv2.FILLED)

        # 2) Warp ambas mascaras al espacio metrico (1 px = 1 cm).
        #    INTER_NEAREST mantiene los valores binarios sin interpolar.
        cm_walls = cv2.warpPerspective(wall_mask, self.H,
                                       (self._cm_w, self._cm_h),
                                       flags=cv2.INTER_NEAREST)
        cm_obst = cv2.warpPerspective(obst_mask, self.H,
                                      (self._cm_w, self._cm_h),
                                      flags=cv2.INTER_NEAREST)

        # 3) Bajar resolucion a celdas. INTER_AREA hace una especie de
        #    promediado: si una celda toca aunque sea un pixel ocupado el
        #    valor sube; con el umbral de 30 (sobre 255) eso es ~12% de
        #    ocupacion -- conservador, marca la celda como ocupada en
        #    cuanto haya algo de pared/obstaculo.
        small_walls = cv2.resize(cm_walls, (self.cols, self.rows),
                                 interpolation=cv2.INTER_AREA)
        small_obst = cv2.resize(cm_obst, (self.cols, self.rows),
                                interpolation=cv2.INTER_AREA)

        grid = np.zeros((self.rows, self.cols), dtype=np.uint8)
        grid[small_obst > 30] = OBSTACLE
        # Paredes prevalecen sobre obstaculos: si la celda es ambos a la vez,
        # se marca como WALL (mismo efecto para A* pero mejor visualizacion).
        grid[small_walls > 30] = WALL

        # Borde virtual del campo: las paredes externas de la pista quedan
        # FUERA del rectangulo navegable (la homografia se calibro con las
        # esquinas internas). Para que la inflacion del robot mantenga
        # distancia del perimetro, marcamos la celda exterior como WALL.
        grid[0, :] = WALL
        grid[-1, :] = WALL
        grid[:, 0] = WALL
        grid[:, -1] = WALL

        # 4) Inflar por el radio del robot. Las celdas que se "ganan" al
        #    dilatar (estaban libres y ahora estan junto a un obstaculo)
        #    se marcan como INFLATED. A* las trata como bloqueadas pero
        #    se distinguen visualmente.
        if self.robot_radius_cm > 0:
            r_cells = max(1, int(np.ceil(self.robot_radius_cm / self.cell_cm)))
            kernel = np.ones((2 * r_cells + 1, 2 * r_cells + 1), np.uint8)
            occupied = (grid > 0).astype(np.uint8) * 255
            dilated = cv2.dilate(occupied, kernel)
            inflated_mask = (dilated > 0) & (grid == FREE)
            grid[inflated_mask] = INFLATED

        return grid

    # ----------------------------------------------------------------- helpers
    def is_blocked(self, grid, row, col):
        """True si la celda esta bloqueada para A* (no FREE)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return True
        return grid[row, col] != FREE

    def cm_to_cell(self, x_cm, y_cm):
        """Convierte (x_cm, y_cm) -> (row, col). None si fuera del campo."""
        col = int(x_cm / self.cell_cm)
        row = int(y_cm / self.cell_cm)
        if not (0 <= col < self.cols and 0 <= row < self.rows):
            return None
        return (row, col)

    def cell_to_cm(self, row, col):
        """Devuelve (x_cm, y_cm) en el centro de la celda."""
        x = (col + 0.5) * self.cell_cm
        y = (row + 0.5) * self.cell_cm
        return (x, y)

    # ----------------------------------------------------------------- render
    def render(self, grid, color_dets=None, scale=6):
        """Dibuja una imagen del grid lista para cv2.imshow.

        scale: pixeles por celda. Con cell_cm=5 y scale=6 una celda son
               6 px en pantalla = 30 px por cada 25 cm.
        color_dets: opcional, dict color -> lista detecciones (con 'pos_cm')
                    para superponer cubos y zonas.
        """
        h_img = self.rows * scale
        w_img = self.cols * scale

        # Fondo blanco roto, no puro blanco, para que se vea bien la cuadricula
        img = np.full((h_img, w_img, 3), 245, dtype=np.uint8)

        # Pintar celdas no-libres como rectangulos solidos. Iteramos solo
        # las celdas no-FREE para no recorrer todo el grid en Python.
        palette = {
            INFLATED: (220, 235, 255),   # rosa-azul muy claro
            OBSTACLE: (90,  90,  90),    # gris oscuro
            WALL:     (140, 80,  140),   # magenta apagado
        }
        # Pintamos en orden creciente de "importancia" para que el ultimo
        # tape al anterior: INFLATED -> OBSTACLE -> WALL.
        for value in (INFLATED, OBSTACLE, WALL):
            ys, xs = np.where(grid == value)
            color = palette[value]
            for r, c in zip(ys, xs):
                cv2.rectangle(img, (c * scale, r * scale),
                              ((c + 1) * scale, (r + 1) * scale),
                              color, thickness=cv2.FILLED)

        # Cuadricula tenue cada 50 cm para tener referencia rapida.
        step_cells = int(round(50 / self.cell_cm))
        for c in range(0, self.cols + 1, step_cells):
            x = c * scale
            cv2.line(img, (x, 0), (x, h_img - 1), (200, 200, 200), 1)
        for r in range(0, self.rows + 1, step_cells):
            y = r * scale
            cv2.line(img, (0, y), (w_img - 1, y), (200, 200, 200), 1)

        # Marco exterior del campo
        cv2.rectangle(img, (0, 0), (w_img - 1, h_img - 1), (50, 50, 50), 2)

        # Superponer cubos y zonas
        if color_dets:
            color_bgr = {
                'RED':   (0,   0,   220),
                'GREEN': (0,   180, 0),
                'BLUE':  (220, 80,  0),
            }
            for color, dets in color_dets.items():
                bgr = color_bgr.get(color, (0, 0, 0))
                for d in dets:
                    pos = d.get('pos_cm')
                    if pos is None:
                        continue
                    x_cm, y_cm = pos
                    if not (0 <= x_cm <= self.field_w_cm
                            and 0 <= y_cm <= self.field_h_cm):
                        continue
                    px = int(x_cm / self.cell_cm * scale)
                    py = int(y_cm / self.cell_cm * scale)
                    if d['kind'] == 'CUBE':
                        # Cuadrado relleno
                        s = max(4, int(15 / self.cell_cm * scale / 2))
                        cv2.rectangle(img, (px - s, py - s), (px + s, py + s),
                                      bgr, thickness=cv2.FILLED)
                        cv2.rectangle(img, (px - s, py - s), (px + s, py + s),
                                      (0, 0, 0), 1)
                    elif d['kind'] == 'ZONE':
                        s = max(6, int(20 / self.cell_cm * scale / 2))
                        cv2.rectangle(img, (px - s, py - s), (px + s, py + s),
                                      bgr, 2)
                        cv2.line(img, (px - s, py - s), (px + s, py + s),
                                 bgr, 1)
                        cv2.line(img, (px + s, py - s), (px - s, py + s),
                                 bgr, 1)

        # Leyenda arriba a la izquierda
        legend = [
            ("PARED",      (140, 80,  140)),
            ("OBSTACULO",  (90,  90,  90)),
            ("MARGEN",     (220, 235, 255)),
        ]
        y0 = 14
        for label, bgr in legend:
            cv2.rectangle(img, (8, y0 - 8), (22, y0 + 4), bgr,
                          thickness=cv2.FILLED)
            cv2.rectangle(img, (8, y0 - 8), (22, y0 + 4), (0, 0, 0), 1)
            cv2.putText(img, label, (28, y0 + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1,
                        cv2.LINE_AA)
            y0 += 16

        return img

    def draw_path(self, grid_img, path, scale=None, goal=None):
        """Dibuja una ruta de celdas sobre la imagen del grid.

        path: lista de (row, col) devuelta por astar().
        scale: pixeles por celda. Si None, se infiere de grid_img.shape.
        goal: (row, col) del objetivo a marcar con un circulo, opcional.
        """
        if scale is None:
            scale = grid_img.shape[0] / self.rows
        if path:
            pts = np.array(
                [(int((c + 0.5) * scale), int((r + 0.5) * scale))
                 for (r, c) in path], dtype=np.int32)
            cv2.polylines(grid_img, [pts.reshape(-1, 1, 2)],
                          False, (0, 200, 0), 2, cv2.LINE_AA)
            # Waypoints intermedios como puntos pequenos
            for (x, y) in pts[1:-1]:
                cv2.circle(grid_img, (int(x), int(y)), 2, (0, 120, 0), -1)
        if goal is not None:
            gx = int((goal[1] + 0.5) * scale)
            gy = int((goal[0] + 0.5) * scale)
            cv2.drawMarker(grid_img, (gx, gy), (0, 0, 220),
                           cv2.MARKER_CROSS, 16, 2)
            cv2.circle(grid_img, (gx, gy), 8, (0, 0, 220), 2)
        return grid_img

    def stats(self, grid):
        """Devuelve dict con cuentas de cada tipo de celda y % libres."""
        total = grid.size
        n_wall = int(np.sum(grid == WALL))
        n_obst = int(np.sum(grid == OBSTACLE))
        n_infl = int(np.sum(grid == INFLATED))
        n_free = total - n_wall - n_obst - n_infl
        return {
            'total':    total,
            'free':     n_free,
            'wall':     n_wall,
            'obstacle': n_obst,
            'inflated': n_infl,
            'pct_free': 100.0 * n_free / total,
        }
