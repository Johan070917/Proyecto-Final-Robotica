"""
pathfinding.py - A* sobre el occupancy grid.

Recibe el grid de OccupancyGrid (numpy array con FREE/WALL/OBSTACLE/INFLATED)
y devuelve una ruta de celdas desde start a goal.

Diseno:
  - Conectividad 8 (4 rectas + 4 diagonales)
  - Coste recto = 1, coste diagonal = sqrt(2)
  - Heuristica octile (admisible y ajustada para 8-conectado)
  - Anti-corner-cutting: no permitir diagonal si una de las celdas
    adyacentes que toca la diagonal esta bloqueada (evita "atravesar"
    paredes finas)
  - Si el start cae sobre una celda bloqueada (puede pasar: el ArUco
    devuelve la pose del marcador pero el robot ocupa varias celdas y
    alguna puede haber sido inflada), se busca la celda libre mas
    cercana como start efectivo.

Uso:
    path_cells = astar(occupancy, start_cell, goal_cell)
    path_cm    = [grid_builder.cell_to_cm(r, c) for (r, c) in path_cells]
"""

import heapq
import math

import cv2
import numpy as np

from occupancy_grid import FREE, INFLATED


_SQRT2 = math.sqrt(2.0)

# Coste de entrar en una celda = 1 + penalizacion por cercania a pared.
# La penalizacion no es binaria: usa un campo de holgura (distancia en
# celdas a la pared/obstaculo mas cercano) para que el coste BAJE cuanto
# mas lejos de una pared estes. Eso crea un "valle" de coste minimo por
# el centro de los pasillos, que es justo donde el robot debe ir para
# alinearse con la cinta del seguidor de linea.
#
#   clear >= CLEARANCE_TARGET_CELLS  -> penalizacion 0 (espacio abierto)
#   clear -> 0 (pegado a la pared)   -> penalizacion WALL_PENALTY
#
# Subir WALL_PENALTY = el robot se centra mas (acepta rodeos para
# despegarse de paredes). Subir CLEARANCE_TARGET_CELLS = el gradiente
# llega mas lejos de la pared (centra mejor en pasillos anchos).
WALL_PENALTY           = 5.0
CLEARANCE_TARGET_CELLS = 5.0

# 8 vecinos: (dr, dc, coste). Los 4 primeros son rectos (coste 1), los 4
# ultimos diagonales (coste sqrt(2)).
_NEIGHBORS = [
    (-1,  0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, _SQRT2), (-1, 1, _SQRT2),
    ( 1, -1, _SQRT2), ( 1, 1, _SQRT2),
]


def _octile(a, b):
    """Distancia octile: heuristica exacta para 8-conectado sin obstaculos."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return max(dr, dc) + (_SQRT2 - 1.0) * min(dr, dc)


def _is_traversable(occ, r, c):
    """Una celda es transitable si NO es pared/obstaculo. INFLATED se
    permite pagando un coste extra (ver _enter_cost)."""
    rows, cols = occ.shape
    if not (0 <= r < rows and 0 <= c < cols):
        return False
    return occ[r, c] in (FREE, INFLATED)


def clearance_field(occ):
    """Distancia (en celdas) de cada celda a la pared/obstaculo mas cercano.

    Las celdas transitables (FREE/INFLATED) son foreground; WALL/OBSTACLE
    son las fuentes (distancia 0). Una celda pegada a una pared sale ~1;
    el centro de un pasillo de media-anchura 3 celdas sale ~3.

    Se calcula una vez por plan (el grid es ~80x42, microsegundos) y se
    pasa a astar() para no recomputarlo en cada llamada greedy.
    """
    traversable = ((occ == FREE) | (occ == INFLATED)).astype(np.uint8)
    # distanceTransform: distancia de cada pixel !=0 al 0 mas cercano.
    return cv2.distanceTransform(traversable, cv2.DIST_L2, 3)


def _enter_cost(clearance, r, c):
    """Multiplicador del coste de entrar en (r, c).

    1.0 en espacio abierto; sube suavemente hasta 1 + WALL_PENALTY al
    pegarse a una pared. Gradiente continuo => A* prefiere el centro.
    """
    clear = float(clearance[r, c])
    t = min(1.0, clear / CLEARANCE_TARGET_CELLS)   # 0 pared .. 1 abierto
    return 1.0 + WALL_PENALTY * (1.0 - t)


def _nearest_traversable(occ, cell, max_radius=20):
    """Devuelve la celda transitable (FREE o INFLATED) mas cercana a 'cell'.

    Util cuando start o goal cae sobre una pared/obstaculo. No relocaliza
    cuando ya es transitable.
    """
    if _is_traversable(occ, cell[0], cell[1]):
        return cell
    rows, cols = occ.shape
    seen = {cell}
    frontier = [cell]
    for _ in range(max_radius):
        next_frontier = []
        for r, c in frontier:
            for dr, dc, _ in _NEIGHBORS:
                nb = (r + dr, c + dc)
                if nb in seen:
                    continue
                seen.add(nb)
                if not (0 <= nb[0] < rows and 0 <= nb[1] < cols):
                    continue
                if _is_traversable(occ, nb[0], nb[1]):
                    return nb
                next_frontier.append(nb)
        if not next_frontier:
            return None
        frontier = next_frontier
    return None


def astar(occ, start, goal, allow_relax=True, clearance=None):
    """A* en grid 8-conectado.

    Args:
        occ:   numpy array (rows, cols) con el grid de ocupacion.
        start: (row, col) inicial.
        goal:  (row, col) objetivo.
        allow_relax: si start o goal caen sobre celda bloqueada, busca la
                     celda libre mas cercana en vez de fallar.
        clearance: campo de holgura precalculado (clearance_field). Si es
                   None se calcula aqui. Pasarlo cuando se llama a astar
                   muchas veces sobre el mismo grid (planificador greedy).

    Devuelve:
        Lista de celdas [(r0,c0), (r1,c1), ...] incluyendo start y goal,
        o None si no hay ruta.
    """
    rows, cols = occ.shape
    if clearance is None:
        clearance = clearance_field(occ)
    if allow_relax:
        s = _nearest_traversable(occ, start)
        g = _nearest_traversable(occ, goal)
        if s is None or g is None:
            return None
        start, goal = s, g
    else:
        if (not _is_traversable(occ, *start)
                or not _is_traversable(occ, *goal)):
            return None

    if start == goal:
        return [start]

    # open set: heap de (f, counter, cell). counter evita comparar tuplas
    # de celdas cuando f empata.
    open_heap = []
    counter = 0
    heapq.heappush(open_heap, (_octile(start, goal), counter, start))

    came_from = {}
    g_score = {start: 0.0}
    closed = np.zeros((rows, cols), dtype=bool)

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if closed[current]:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed[current] = True
        cr, cc = current

        for dr, dc, step_cost in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if closed[nr, nc]:
                continue
            if not _is_traversable(occ, nr, nc):
                continue
            # Anti-corner-cutting: no permitir cruzar en diagonal si una
            # de las celdas adyacentes (que comparte arista con la diagonal)
            # esta bloqueada por pared/obstaculo. INFLATED si se permite.
            if dr != 0 and dc != 0:
                if (not _is_traversable(occ, cr + dr, cc)
                        or not _is_traversable(occ, cr, cc + dc)):
                    continue

            # Coste de entrar en la celda: gradiente por cercania a pared
            # (mas barato cuanto mas centrado), ver _enter_cost.
            tentative = (g_score[current]
                         + step_cost * _enter_cost(clearance, nr, nc))
            if tentative < g_score.get((nr, nc), float('inf')):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative
                f = tentative + _octile((nr, nc), goal)
                counter += 1
                heapq.heappush(open_heap, (f, counter, (nr, nc)))

    return None


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def simplify_path(path):
    """Quita celdas intermedias en tramos rectos: ABCDE en linea -> AE.

    No es line-of-sight smoothing; solo elimina puntos colineales del path
    de A*. Suficiente para que el robot reciba pocos waypoints en lugar de
    una celda por paso.
    """
    if path is None or len(path) <= 2:
        return path
    out = [path[0]]
    prev_dr, prev_dc = None, None
    for i in range(1, len(path)):
        dr = path[i][0] - path[i - 1][0]
        dc = path[i][1] - path[i - 1][1]
        if (dr, dc) != (prev_dr, prev_dc):
            if i > 1:
                out.append(path[i - 1])
            prev_dr, prev_dc = dr, dc
    out.append(path[-1])
    return out


def path_length_cm(path, cell_cm):
    """Longitud aproximada del path en cm (suma de pasos)."""
    if path is None or len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path, path[1:]):
        dr = abs(a[0] - b[0])
        dc = abs(a[1] - b[1])
        step = _SQRT2 if (dr and dc) else 1.0
        total += step
    return total * cell_cm
