"""
mission.py - Planificacion de mision completa.

Toma:
  - Posicion del robot (cm)
  - Detecciones de cubos y zonas por color (con sus pos_cm)
  - El occupancy grid + builder
  - El A*

Devuelve una lista ordenada de TAREAS, donde cada tarea es:

    color cubo -> ruta_robot_al_cubo -> ruta_cubo_a_zona

Politica de orden (greedy por cercania):
  1. Estado inicial: robot en su pos_cm actual
  2. De los cubos no visitados, elegir el que tenga la ruta MAS CORTA
     desde la posicion actual (cube_pos)
  3. Despues de "entregar" ese cubo, el robot esta en su zona
     (zone_pos del mismo color)
  4. Repetir hasta agotar cubos

No es optimo global (eso seria TSP), pero con 3-9 cubos da rutas razonables
y se calcula al instante. Si despues queremos optimo, se puede cambiar la
politica sin tocar el resto del codigo.

Asunciones de esta primera version:
  - Los cubos en pantalla son DISTINTOS (no apila aun varios cubos del
    mismo color en una zona; cuando haya 3 cubos rojos visibles los tratara
    como 3 tareas separadas con destino la misma zona, lo cual es correcto
    para apilar). Esto funciona porque cada cubo tiene una posicion fisica
    diferente aunque luego acaben en la misma zona.
  - Se planifica AL CENTRO del cubo y AL CENTRO de la zona. Los sensores
    ToF / finales de carrera del robot se encargan de la aproximacion final.
"""

import cv2
import numpy as np

from pathfinding import astar, simplify_path, path_length_cm


# Colores para dibujar cada paso de la mision en el grid (BGR).
# Se ciclan si hay mas tareas que colores disponibles.
ROUTE_COLORS = [
    (0,   200, 0),    # verde
    (220, 180, 0),    # cian
    (200, 0,   200),  # magenta
    (0,   140, 220),  # naranja
    (180, 180, 0),    # turquesa
    (0,   80,  220),  # rojo-naranja
]


def _collect_cubes_and_zones(color_dets):
    """Separa el dict de detecciones en listas de cubos y un dict de zonas.

    Devuelve (cubes, zones_by_color):
        cubes: lista de dicts con 'color' y 'pos_cm'
        zones_by_color: {'RED': pos_cm o None, 'GREEN': ..., 'BLUE': ...}

    Solo incluye detecciones con pos_cm disponible (necesita homografia).
    """
    cubes = []
    zones_by_color = {'RED': None, 'GREEN': None, 'BLUE': None}
    for color, dets in color_dets.items():
        for d in dets:
            pos = d.get('pos_cm')
            if pos is None:
                continue
            if d['kind'] == 'CUBE':
                cubes.append({'color': color, 'pos_cm': pos})
            elif d['kind'] == 'ZONE':
                # Si hay varias zonas detectadas del mismo color (raro),
                # nos quedamos con la primera.
                if zones_by_color[color] is None:
                    zones_by_color[color] = pos
    return cubes, zones_by_color


def plan(robot_pos_cm, color_dets, grid_builder, occupancy):
    """Genera la lista ordenada de tareas con sus rutas A*.

    Devuelve un dict con:
        'tasks':   lista de tareas en orden de ejecucion
        'skipped': lista de cubos descartados (sin zona del mismo color)
        'total_cm': longitud total estimada de la mision

    Cada tarea es un dict:
        {
            'color': 'RED' / 'GREEN' / 'BLUE',
            'cube_cm':       (x, y),
            'zone_cm':       (x, y),
            'route_to_cube': [(r, c), ...],  # celdas, simplificadas
            'route_to_zone': [(r, c), ...],
            'cm_to_cube':    float,
            'cm_to_zone':    float,
        }
    """
    cubes, zones = _collect_cubes_and_zones(color_dets)

    # Quitar cubos cuya zona no se ve: no sabemos donde dejarlos
    valid = []
    skipped = []
    for cube in cubes:
        if zones[cube['color']] is None:
            skipped.append(cube)
        else:
            valid.append(cube)

    tasks = []
    total_cm = 0.0
    current_cm = robot_pos_cm
    remaining = list(valid)

    while remaining:
        best = None
        best_route = None
        best_dist = float('inf')

        # Para cada cubo restante, calcular ruta desde la posicion actual.
        # Nos quedamos con el cubo cuya ruta sea mas corta.
        start_cell = grid_builder.cm_to_cell(*current_cm)
        if start_cell is None:
            break
        for cube in remaining:
            goal_cell = grid_builder.cm_to_cell(*cube['pos_cm'])
            if goal_cell is None:
                continue
            r = astar(occupancy, start_cell, goal_cell)
            if r is None:
                continue
            d = path_length_cm(r, grid_builder.cell_cm)
            if d < best_dist:
                best_dist = d
                best = cube
                best_route = r

        if best is None:
            # No se pudo llegar a ninguno de los cubos restantes.
            skipped.extend(remaining)
            break

        # Ruta del cubo a su zona
        zone_cm = zones[best['color']]
        cube_cell = grid_builder.cm_to_cell(*best['pos_cm'])
        zone_cell = grid_builder.cm_to_cell(*zone_cm)
        route_to_zone = astar(occupancy, cube_cell, zone_cell)
        if route_to_zone is None:
            # No hay ruta hasta la zona: descartar este cubo y seguir
            skipped.append(best)
            remaining.remove(best)
            continue
        dist_zone = path_length_cm(route_to_zone, grid_builder.cell_cm)

        tasks.append({
            'color':         best['color'],
            'cube_cm':       best['pos_cm'],
            'zone_cm':       zone_cm,
            'route_to_cube': simplify_path(best_route),
            'route_to_zone': simplify_path(route_to_zone),
            'cm_to_cube':    best_dist,
            'cm_to_zone':    dist_zone,
        })
        total_cm += best_dist + dist_zone
        current_cm = zone_cm
        remaining.remove(best)

    return {
        'tasks':    tasks,
        'skipped':  skipped,
        'total_cm': total_cm,
    }


def draw_plan(grid_img, plan_data, grid_builder):
    """Dibuja todas las rutas de la mision sobre la imagen del grid.

    Cada tarea se pinta con un color distinto. Se numeran los waypoints
    para que se vea claramente el orden de ejecucion.
    """
    if plan_data is None or not plan_data['tasks']:
        return grid_img

    scale = grid_img.shape[0] / grid_builder.rows

    def _cells_to_pts(cells):
        return np.array(
            [(int((c + 0.5) * scale), int((r + 0.5) * scale))
             for (r, c) in cells], dtype=np.int32)

    for i, task in enumerate(plan_data['tasks']):
        color = ROUTE_COLORS[i % len(ROUTE_COLORS)]
        # Ruta al cubo: linea continua
        pts1 = _cells_to_pts(task['route_to_cube'])
        if len(pts1) >= 2:
            cv2.polylines(grid_img, [pts1.reshape(-1, 1, 2)],
                          False, color, 2, cv2.LINE_AA)
        # Ruta a la zona: linea discontinua
        pts2 = _cells_to_pts(task['route_to_zone'])
        if len(pts2) >= 2:
            # Dibujar la ruta como segmentos cortos alternos
            for k in range(0, len(pts2) - 1, 2):
                a = tuple(pts2[k])
                b = tuple(pts2[min(k + 1, len(pts2) - 1)])
                cv2.line(grid_img, a, b, color, 2, cv2.LINE_AA)

        # Numero del paso encima del cubo
        cube_px = pts1[-1]
        cv2.circle(grid_img, tuple(cube_px), 12, (255, 255, 255), -1)
        cv2.circle(grid_img, tuple(cube_px), 12, color, 2)
        cv2.putText(grid_img, str(i + 1),
                    (cube_px[0] - 5, cube_px[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    return grid_img


def summary(plan_data):
    """Devuelve un string multi-linea con el resumen de la mision."""
    if plan_data is None:
        return "(sin plan)"
    lines = []
    for i, t in enumerate(plan_data['tasks'], 1):
        cx, cy = t['cube_cm']
        zx, zy = t['zone_cm']
        lines.append(
            f"  {i}. {t['color']:5s}  cubo ({cx:.0f},{cy:.0f}) -> "
            f"zona ({zx:.0f},{zy:.0f})   "
            f"{t['cm_to_cube']:.0f} + {t['cm_to_zone']:.0f} cm")
    if plan_data['skipped']:
        lines.append(
            f"  Descartados: {len(plan_data['skipped'])} "
            f"(sin zona o sin ruta)")
    lines.append(f"  TOTAL: {plan_data['total_cm']:.0f} cm "
                 f"en {len(plan_data['tasks'])} tareas")
    return "\n".join(lines)
