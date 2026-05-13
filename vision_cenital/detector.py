"""
detector.py - Detector cenital para Proyecto Final Robotica.

Corre en el PORTATIL (Windows), NO en la RPi.
Lee la camara cenital del techo y detecta:
  - Cubos R/G/B (cuadrado lleno de color)
  - Zonas de entrega R/G/B (cuadrado con X de cinta)
  - Obstaculos (blobs negros, excepto la cinta-linea del puente)

Esta version NO publica nada todavia -- solo muestra ventana de debug.
La integracion con la RPi (socket TCP/JSON) se anade en una segunda iteracion
una vez calibrada la deteccion con tu camara real.

Uso tipico:

    # Detectar en vivo con la webcam (camara 0)
    python detector.py

    # Probar sobre una foto guardada (e.g. la del piso azul)
    python detector.py --image zona_azul.jpg

    # Calibrar rangos HSV con trackbars (mientras enfocas un objeto del color)
    python detector.py --tune BLUE

Teclas dentro de la ventana:
    q  salir
    s  guardar snapshot de el frame actual
    h  esconder/mostrar overlay
    g  esconder/mostrar ventana de grid de ocupacion
"""

import argparse
import json
import os
import threading
import time

# Desactiva las "HW transforms" de MSMF antes de importar cv2. Sin esto,
# probar un indice de camara inexistente con MSMF en Windows cuelga sin
# timeout. Tiene que estar ANTES de "import cv2".
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from occupancy_grid import OccupancyGrid  # noqa: E402

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False


CAMERA_PARAMS_FILE = os.path.join(os.path.dirname(__file__),
                                  'camera_params.json')
HOMOGRAPHY_FILE    = os.path.join(os.path.dirname(__file__),
                                  'homography.json')


class Undistorter:
    """Aplica correccion de distorsion radial si hay camera_params.json.

    Si no existe el archivo, pasa los frames sin modificar. Esto permite
    usar el detector antes de calibrar y mejorarlo despues sin tocar nada.
    """

    def __init__(self):
        self.enabled = False
        self.K = None
        self.dist = None
        self.new_K = None
        self.map1 = None
        self.map2 = None
        self._size = None
        if os.path.exists(CAMERA_PARAMS_FILE):
            try:
                with open(CAMERA_PARAMS_FILE) as fh:
                    data = json.load(fh)
                self.K = np.array(data['camera_matrix'], dtype=np.float64)
                self.dist = np.array(data['distortion_coeffs'],
                                     dtype=np.float64)
                self.enabled = True
                print(f"[undistort] Cargado {CAMERA_PARAMS_FILE} "
                      f"(RMS calibracion: {data.get('rms', '?')})")
            except Exception as exc:
                print(f"[undistort] No pude cargar params: {exc}")

    def __call__(self, frame):
        if not self.enabled:
            return frame
        h, w = frame.shape[:2]
        if self._size != (w, h):
            # alpha=0: recorta los bordes curvos negros que crea undistort.
            # Sin esto, esas zonas negras forman un contorno gigante que
            # envuelve todo el campo y oculta los obstaculos al usar
            # RETR_EXTERNAL. Se pierde un poco de campo de vision en las
            # esquinas pero la imagen queda limpia.
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.dist, (w, h), 0, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2)
            self._size = (w, h)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


class Homography:
    """Convierte puntos de imagen (px, py) a coordenadas reales (x_cm, y_cm).

    Carga homography.json si existe. Si no, queda deshabilitado y el detector
    funciona igual pero sin coordenadas mundiales.
    """

    def __init__(self):
        self.enabled = False
        self.H = None
        self.field_w_cm = None
        self.field_h_cm = None
        if os.path.exists(HOMOGRAPHY_FILE):
            try:
                with open(HOMOGRAPHY_FILE) as fh:
                    data = json.load(fh)
                self.H = np.array(data['homography'], dtype=np.float64)
                self.field_w_cm = float(data['field_width_cm'])
                self.field_h_cm = float(data['field_height_cm'])
                self.enabled = True
                print(f"[homography] Cargado {HOMOGRAPHY_FILE} "
                      f"(campo {self.field_w_cm:.0f}x{self.field_h_cm:.0f} cm)")
            except Exception as exc:
                print(f"[homography] No pude cargar: {exc}")

    def to_cm(self, points):
        """Convierte una lista de (px, py) a lista de (x_cm, y_cm).

        Devuelve None si no esta habilitado o la lista es vacia.
        """
        if not self.enabled or not points:
            return None
        pts = np.array([[p] for p in points], dtype=np.float32)
        out = cv2.perspectiveTransform(pts, self.H)
        return [(float(p[0, 0]), float(p[0, 1])) for p in out]


class LatestFrameReader:
    """Lee la camara en un hilo separado y expone siempre el frame mas reciente.

    Sin esto, cap.read() bloquea el hilo de procesamiento esperando el proximo
    frame USB. Si el procesamiento tarda mas que 1/fps, se acumula un retraso
    visible. Con este lector, el hilo de procesamiento siempre coge el frame
    mas reciente disponible sin esperar.
    """

    def __init__(self, cap):
        self._cap = cap
        self._frame = None
        self._lock = threading.Lock()
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def _loop(self):
        while self._running:
            ok, frame = self._cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self):
        self._running = False
        self._t.join(timeout=2.0)


# -----------------------------------------------------------------------------
# Rangos HSV iniciales (calibrar con --tune)
# -----------------------------------------------------------------------------
HSV_RANGES = {
    'RED':   [((0,   110,  80), (10,  255, 255)),
              ((170, 110,  80), (180, 255, 255))],
    'GREEN': [((34,   26,  40), (86,  255, 255))],
    'BLUE':  [((90,   37,  40), (130, 255, 255))],
}

# Negro para obstaculos: cualquier H, S baja-media, V baja.
HSV_BLACK = ((0, 0, 0), (180, 80, 70))


# -----------------------------------------------------------------------------
# Parametros de deteccion (calibrar segun resolucion de tu camara)
# -----------------------------------------------------------------------------
# A 1280x720 sobre pista 2x2 m, 1 px ~= 0.16 cm. Un cubo de 15 cm ocupa
# unos 90 px de lado = ~8000 px de area. Una zona ocupa lo mismo pero con
# fill-ratio bajo. Bajamos los umbrales para que sea robusto incluso si
# la camara baja a 480p.
MIN_AREA_COLOR = 400       # area minima del blob de color (px)
MIN_AREA_OBST  = 300       # area minima de obstaculo
ASPECT_TOL     = (0.45, 2.20)  # tolerancia para cubos / zonas

# Clasificacion CUBE / ZONE:
#   - El fill-ratio (mask_count / bbox_area) falla cuando el cubo tiene
#     brillos/sombras que rompen la mascara HSV (un cubo puede dar fill 0.45
#     y confundirse con una zona).
#   - Mejor test: erosionar la mascara. Un cubo solido sobrevive (queda un
#     "nucleo" central). Una zona con X de cinta delgada se borra entera.
FILL_ZONE_MAX       = 0.65     # fill maximo para ser ZONE
CORE_FILL_CUBE_MIN  = 0.05     # >= 5% del bbox sobrevive a la erosion -> CUBE
ERODE_KERNEL_REL    = 0.20     # erosion del 20% del lado del bbox

# Filtros de obstaculos / paredes (mismo mask negro).
MIN_AREA_WALL      = 800   # area minima para considerarlo pared
TAPE_MIN_DIM       = 6     # min dim < esto = ultra delgado, descartar siempre
# Cinta-linea del puente (line follower): alargada Y delgada. Las paredes
# reales son notablemente mas gruesas, asi que filtramos cualquier tira
# alargada cuyo grosor sea menor que TAPE_MAX_THICK.
TAPE_ASPECT_THIN   = 3.0   # aspect > este
TAPE_MAX_THICK     = 20    # AND min_dim < este -> es cinta, descartar
OBST_ASPECT_MAX    = 2.5   # un obstaculo (cubo negro) tiene aspect ~1; 2.5 tolera sombras
OBST_EXTENT_MIN    = 0.35  # extent (area/bbox) minimo para obstaculo compacto
MAX_OBST_AREA      = 20000 # area maxima para obstaculo -- mas grande = pared


# -----------------------------------------------------------------------------
# Colores de dibujo (BGR)
# -----------------------------------------------------------------------------
DRAW_BGR = {'RED': (0, 0, 255), 'GREEN': (0, 220, 0), 'BLUE': (255, 80, 0)}


# -----------------------------------------------------------------------------
def _mask_for_color(hsv, ranges):
    """Devuelve mascara binaria con la union de los rangos HSV del color."""
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                            np.array(hi, dtype=np.uint8))
    return mask


def detect_color_objects(hsv, color_name, ranges):
    """Detecta blobs del color dado y los clasifica en CUBE / ZONE / UNKNOWN.

    Retorna lista de dicts con bbox, kind, fill_ratio, contorno y area.
    """
    raw = _mask_for_color(hsv, ranges)

    # Suavizar
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Cerrar fuerte para que la X de la zona se vuelva un cuadrado solido,
    # asi findContours nos da UN contorno por zona (igual que por cubo).
    # Kernel escala con la resolucion: 2.5% del alto del frame.
    h_img = hsv.shape[0]
    k_size = max(7, int(h_img * 0.025))
    k = np.ones((k_size, k_size), np.uint8)
    closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)

    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA_COLOR:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = w / float(h)
        if not (ASPECT_TOL[0] <= aspect <= ASPECT_TOL[1]):
            continue

        # Fill ratio en la MASCARA CRUDA (sin cerrar) dentro del bbox.
        # Cubo lleno -> ~1.0   |   Zona con X -> ~0.3-0.5
        roi = raw[y:y + h, x:x + w]
        bbox_area = float(w * h)
        fill = cv2.countNonZero(roi) / bbox_area

        # Test de erosion: si erosionamos la mascara cruda en el bbox y queda
        # un "nucleo", es porque hay una region solida (cubo). Si no queda
        # nada, eran lineas delgadas (zona con X de cinta).
        erode_k = max(3, int(min(w, h) * ERODE_KERNEL_REL))
        roi_eroded = cv2.erode(roi, np.ones((erode_k, erode_k), np.uint8))
        core_fill = cv2.countNonZero(roi_eroded) / bbox_area

        if core_fill >= CORE_FILL_CUBE_MIN:
            kind = 'CUBE'
        elif fill <= FILL_ZONE_MAX:
            kind = 'ZONE'
        else:
            kind = 'UNKNOWN'

        cx = x + w // 2
        cy = y + h // 2
        out.append({
            'color': color_name,
            'kind': kind,
            'bbox': (x, y, w, h),
            'center': (cx, cy),
            'area': area,
            'fill': fill,
            'contour': c,
        })
    return out


def detect_black_structures(hsv):
    """Detecta todo lo negro y lo clasifica en obstaculos discretos vs paredes.

    Un mismo blob negro puede ser:
      - Cinta-linea del puente (muy delgada en algun eje) -> DESCARTADO
      - Obstaculo compacto tipo cubo (aspect ~1, area moderada) -> obstacle
      - Pared (alargada o muy grande) -> wall

    Devuelve (obstacles, walls, wall_mask). wall_mask sirve luego para
    construir el grid de ocupacion para A*/BFS/DFS.
    """
    lo, hi = HSV_BLACK
    raw_mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                           np.array(hi, dtype=np.uint8))
    # OPEN fuerte: elimina componentes pequenos antes del CLOSE. Esto borra
    # los huecos individuales de la rejilla metalica (4-8 px cada uno) para
    # que el CLOSE posterior NO los una al contorno de la pared cercana.
    # Las paredes reales (>= 20 px de grosor) no se ven afectadas: OPEN no
    # encoge componentes grandes, solo elimina los que caben dentro del kernel.
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,
                                np.ones((7, 7), np.uint8))
    # Cerrar para unir paredes con pequenos huecos.
    # Kernel 5x5 (no 7x7): la cinta-linea del puente queda a ~13 px de grosor
    # en vez de ~15 px, dando mas margen al filtro TAPE_MAX_THICK.
    clean = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE,
                             np.ones((5, 5), np.uint8))

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    obstacles, walls = [], []
    wall_mask = np.zeros_like(clean)

    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA_OBST:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        min_dim = min(w, h)
        aspect = max(w, h) / float(min_dim)
        extent = area / float(w * h)

        # Cinta-linea del puente: descartar.
        #  (a) ultra delgada en algun eje
        #  (b) alargada Y mas fina que TAPE_MAX_THICK
        if min_dim < TAPE_MIN_DIM:
            continue
        if aspect > TAPE_ASPECT_THIN and min_dim < TAPE_MAX_THICK:
            continue

        # Obstaculo compacto: aspect cercano a 1, extent alto, area moderada.
        is_compact = (aspect <= OBST_ASPECT_MAX
                      and extent >= OBST_EXTENT_MIN
                      and area <= MAX_OBST_AREA)
        if is_compact:
            obstacles.append({
                'bbox': (x, y, w, h),
                'center': (x + w // 2, y + h // 2),
                'area': area,
            })
        elif area >= MIN_AREA_WALL or aspect > OBST_ASPECT_MAX:
            walls.append({
                'bbox': (x, y, w, h),
                'area': area,
                'contour': c,
            })
            cv2.drawContours(wall_mask, [c], -1, 255, thickness=cv2.FILLED)

    return obstacles, walls, wall_mask


# -----------------------------------------------------------------------------
def _cm_label(d):
    """Devuelve string '(x,y)cm' si la deteccion tiene pos_cm, si no ''."""
    if 'pos_cm' in d and d['pos_cm'] is not None:
        x_cm, y_cm = d['pos_cm']
        return f"({x_cm:.0f},{y_cm:.0f})cm"
    return ""


def annotate(frame, color_dets, obstacles, walls=None):
    # Dibujar paredes primero (debajo del resto)
    if walls:
        for wl in walls:
            cv2.drawContours(frame, [wl['contour']], -1, (200, 100, 200), 2)
        # Etiqueta solo a la pared mas grande para no llenar de texto
        biggest = max(walls, key=lambda w: w['area'])
        x, y, w, h = biggest['bbox']
        cv2.putText(frame, f"WALL x{len(walls)}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 200), 2)

    for color, dets in color_dets.items():
        bgr = DRAW_BGR[color]
        for d in dets:
            x, y, w, h = d['bbox']
            thick = 3 if d['kind'] == 'CUBE' else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, thick)
            # Marca de tipo: relleno si CUBE, X si ZONE
            if d['kind'] == 'ZONE':
                cv2.line(frame, (x, y), (x + w, y + h), bgr, 1)
                cv2.line(frame, (x + w, y), (x, y + h), bgr, 1)
            label = f"{color} {d['kind']} f={d['fill']:.2f}"
            cv2.putText(frame, label, (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)
            cm_lbl = _cm_label(d)
            if cm_lbl:
                cv2.putText(frame, cm_lbl, (x, y + h + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1,
                            cv2.LINE_AA)

    for o in obstacles:
        x, y, w, h = o['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 2)
        cv2.putText(frame, "OBST", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 2)
        cm_lbl = _cm_label(o)
        if cm_lbl:
            cv2.putText(frame, cm_lbl, (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1,
                        cv2.LINE_AA)
    return frame


def _bbox_overlap_ratio(a, b):
    """Devuelve el ratio de interseccion respecto al area MENOR de los dos
    bboxes. 1.0 = uno contenido completamente en el otro; 0.0 = no se tocan.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(ax, bx)
    iy = max(ay, by)
    ex = min(ax + aw, bx + bw)
    ey = min(ay + ah, by + bh)
    if ex <= ix or ey <= iy:
        return 0.0
    inter = (ex - ix) * (ey - iy)
    return inter / max(1.0, min(aw * ah, bw * bh))


# Si un obstaculo solapa mas que este ratio con un objeto de color, es un
# falso positivo (sombra sobre un cubo, etc.) y se descarta.
OBST_OVERLAP_THRESHOLD = 0.3


def attach_world_coords(homography, color_dets, obstacles, walls):
    """Anade el campo 'pos_cm' a cada deteccion con su posicion mundial.

    Si la homografia no esta habilitada, no hace nada.
    """
    if not homography.enabled:
        return

    targets = []  # lista de (det_dict, center_px)
    for dets in color_dets.values():
        for d in dets:
            targets.append((d, d['center']))
    for o in obstacles:
        targets.append((o, o['center']))
    for w in walls:
        x, y, ww, wh = w['bbox']
        w['center'] = (x + ww // 2, y + wh // 2)
        targets.append((w, w['center']))

    if not targets:
        return
    centers_px = [c for _, c in targets]
    cms = homography.to_cm(centers_px)
    for (d, _), cm in zip(targets, cms):
        d['pos_cm'] = cm


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_dets = {
        c: detect_color_objects(hsv, c, r) for c, r in HSV_RANGES.items()
    }
    obstacles, walls, wall_mask = detect_black_structures(hsv)

    # Descartar obstaculos que solapan significativamente con un cubo o zona.
    # Las sombras / reflejos oscuros sobre los cubos disparan el detector de
    # negro y aparecen como falsos OBST.
    color_bboxes = [d['bbox']
                    for dets in color_dets.values() for d in dets]
    obstacles = [
        o for o in obstacles
        if not any(_bbox_overlap_ratio(o['bbox'], cb) > OBST_OVERLAP_THRESHOLD
                   for cb in color_bboxes)
    ]
    return color_dets, obstacles, walls, wall_mask


# -----------------------------------------------------------------------------
# Modo --tune: trackbars para calibrar rangos HSV en vivo
# -----------------------------------------------------------------------------
def run_tuner(cap, color):
    if color not in HSV_RANGES:
        print(f"Color {color} no esta en HSV_RANGES. Usa RED, GREEN o BLUE.")
        return
    # Para RED hay 2 rangos, usamos solo el primero en el tuner
    lo, hi = HSV_RANGES[color][0]
    win = f"Tune {color}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    for name, val, mx in [('H_lo', lo[0], 180), ('S_lo', lo[1], 255),
                          ('V_lo', lo[2], 255), ('H_hi', hi[0], 180),
                          ('S_hi', hi[1], 255), ('V_hi', hi[2], 255)]:
        cv2.createTrackbar(name, win, val, mx, lambda v: None)
    print(f"Ajusta los trackbars hasta que solo se vea {color}.")
    print(f"Haz clic en la ventana de imagen y pulsa q para salir (o cierra "
          f"con la X).")
    h_lo = s_lo = v_lo = h_hi = s_hi = v_hi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_lo = cv2.getTrackbarPos('H_lo', win)
        s_lo = cv2.getTrackbarPos('S_lo', win)
        v_lo = cv2.getTrackbarPos('V_lo', win)
        h_hi = cv2.getTrackbarPos('H_hi', win)
        s_hi = cv2.getTrackbarPos('S_hi', win)
        v_hi = cv2.getTrackbarPos('V_hi', win)
        mask = cv2.inRange(hsv, np.array([h_lo, s_lo, v_lo]),
                           np.array([h_hi, s_hi, v_hi]))
        vis = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow(win, np.hstack(
            [frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), vis]))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        # Detectar cierre por X (la propiedad WND_PROP_VISIBLE devuelve <1)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    print(f"\nRango final para {color}:")
    print(f"  lo=({h_lo},{s_lo},{v_lo})  hi=({h_hi},{s_hi},{v_hi})")
    print(f"\nLinea para pegar en HSV_RANGES (detector.py):")
    print(f"    '{color}': [(({h_lo}, {s_lo}, {v_lo}), "
          f"({h_hi}, {s_hi}, {v_hi}))],")
    cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
BACKENDS = {
    'msmf':  cv2.CAP_MSMF,      # Media Foundation -- mejor para USB en Win11
    'dshow': cv2.CAP_DSHOW,     # DirectShow -- legacy, cuelga con algunas USB
    'any':   cv2.CAP_ANY,       # Que OpenCV elija
}


def _fourcc_to_str(fcc_int):
    fcc_int = int(fcc_int)
    if fcc_int <= 0:
        return "????"
    return "".join(chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4))


def find_camera_index_by_name(name_substring):
    """Busca el indice DSHOW de una camara cuyo nombre contenga la subcadena.

    Usa pygrabber (DirectShow). Devuelve -1 si no encuentra coincidencia.
    """
    if not HAS_PYGRABBER:
        return -1
    try:
        names = FilterGraph().get_input_devices()
    except Exception:
        return -1
    needle = name_substring.lower()
    for i, n in enumerate(names):
        if needle in n.lower():
            print(f"[camera_by_name] '{name_substring}' encontrada como "
                  f"'{n}' en indice DSHOW {i}")
            return i
    print(f"[camera_by_name] No encontre camara con '{name_substring}'. "
          f"Camaras detectadas: {names}")
    return -1


def find_msmf_index_for_named_camera(dshow_idx, want_w, want_h, max_probe=4,
                                     prefer_other=False):
    """Busca el indice MSMF que corresponde a la camara hallada en DSHOW.

    Estrategia: probar cada indice MSMF pidiendo (want_w, want_h) MJPG.
    Por defecto prefiere el indice MSMF que coincide con dshow_idx (MSMF y
    DSHOW suelen enumerar en el mismo orden). Si prefer_other=True, prefiere
    el indice que NO coincide -- util cuando MSMF y DSHOW van invertidos.
    """
    preferred = dshow_idx
    if prefer_other:
        # Preferir el primer indice valido distinto de dshow_idx
        others = [i for i in range(max_probe) if i != dshow_idx]
        preferred = others[0] if others else dshow_idx

    print(f"[msmf_probe] Buscando indice MSMF que acepte "
          f"{want_w}x{want_h} (preferencia: MSMF[{preferred}])...")
    fallback = -1
    order = [preferred] + [i for i in range(max_probe) if i != preferred]
    for i in order:
        if i < 0:
            continue
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, want_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, want_h)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok, _ = cap.read()
            cap.release()
            accepts = ok and actual_w >= want_w and actual_h >= want_h
            tag = "OK" if accepts else "no"
            match_str = " (match DSHOW)" if i == dshow_idx else ""
            print(f"  indice MSMF {i}: acepto {actual_w}x{actual_h}  "
                  f"frame_ok={ok}  -> {tag}{match_str}")
            if accepts:
                if i == preferred:
                    return i
                if fallback < 0:
                    fallback = i
        except Exception as exc:
            print(f"  indice MSMF {i}: error {exc}")
    if fallback >= 0:
        print(f"[msmf_probe] -> elegido indice MSMF {fallback} (fallback)")
    else:
        print("[msmf_probe] Ningun indice MSMF acepto la resolucion.")
    return fallback


def open_camera(index, width, height, backend='msmf', fourcc='MJPG', fps=30,
                manual_exposure=True):
    api = BACKENDS.get(backend, cv2.CAP_MSMF)
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        raise SystemExit(
            f"No pude abrir camara index={index} con backend={backend}. "
            f"Prueba otro backend con --backend (msmf|dshow|any).")

    # Orden importa: FOURCC -> resolucion -> FPS.
    # Sin MJPG OpenCV pide YUY2 sin comprimir, que a 720p+ excede el ancho
    # de banda USB de la C920 -> "abre pero no entrega frames".
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)

    # Fijar exposicion / balance de blancos / enfoque para que los HSV no
    # bailen entre frames. Valores tipicos para C920 bajo luz interior.
    if manual_exposure:
        # 0.25 = modo manual en DSHOW; 1 = manual en MSMF. Probamos ambos.
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)        # 2^-6 s ~ 15 ms
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, 0)            # enfoque al infinito (cenital)

    # Diagnostico: imprimir lo que la camara REALMENTE acepto
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camara: pedido {width}x{height} {fourcc} {fps}fps  |  "
          f"acepto {actual_w}x{actual_h} {actual_fcc} {actual_fps:.0f}fps  |  "
          f"backend={backend}")

    # Algunas camaras necesitan descartar los primeros frames despues de
    # cambiar formato. Probar leer hasta 30 frames antes de declarar fallo.
    ok_frame = None
    for attempt in range(30):
        ok, frame = cap.read()
        if ok and frame is not None:
            ok_frame = frame
            if attempt > 0:
                print(f"  (camara estabilizo despues de descartar {attempt} "
                      f"frames)")
            break
        time.sleep(0.05)

    if ok_frame is None:
        cap.release()
        raise SystemExit(
            f"Camara index={index} abrio con backend={backend} pero NO entrega "
            f"frames tras 30 intentos.\n"
            f"  - FOURCC negociado: {actual_fcc} (pediste {fourcc})\n"
            f"  - Si FOURCC != MJPG, la camara rechazo MJPG. Prueba bajar "
            f"resolucion o usar --backend distinto.\n"
            f"  - Tambien revisa: USB 2.0 vs 3.0, hub no alimentado, otra app "
            f"usando la camara.")
    return cap


def list_cameras():
    """Enumera camaras por nombre (DirectShow) y luego prueba abrirlas."""
    # 1) Enumerar por nombre con DirectShow -- esto no cuelga
    if HAS_PYGRABBER:
        try:
            graph = FilterGraph()
            names = graph.get_input_devices()
        except Exception as exc:
            names = None
            print(f"pygrabber fallo: {exc}\n")
        if names is not None:
            print("== Camaras detectadas por DirectShow ==")
            if not names:
                print("  (ninguna)")
            for i, n in enumerate(names):
                print(f"  indice {i}: {n}")
            print()
    else:
        print("pygrabber no instalado. Para enumerar por nombre:\n"
              "    pip install pygrabber\n")
        names = None

    # 2) Probar abrir cada indice con cada backend
    n = len(names) if names else 3
    print(f"== Probando abrir indices 0..{n - 1} con cada backend ==")
    for backend_name, api in BACKENDS.items():
        print(f"\n  Backend {backend_name}:")
        for i in range(n):
            cap = cv2.VideoCapture(i, api)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    h, w = frame.shape[:2]
                    print(f"    camara {i}: OK  ({w}x{h})")
                else:
                    print(f"    camara {i}: abre pero NO entrega frames")
                cap.release()
            else:
                print(f"    camara {i}: no abre")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=None,
                        help='Indice de la camara (0,1,2...). Si lo pasas '
                             'explicitamente, sobrescribe la auto-deteccion '
                             'por nombre. Default: auto-detectar.')
    parser.add_argument('--camera-name', type=str, default='C920',
                        help='Buscar la camara por subcadena del nombre '
                             '(default "C920"). Mas robusto que --camera '
                             'porque el indice cambia entre reconexiones. '
                             'Ignorado si pasas --camera explicito.')
    parser.add_argument('--prefer-other-msmf', action='store_true',
                        help='Invierte la heuristica MSMF: prefiere el indice '
                             'que NO coincide con DSHOW. Util si la '
                             'auto-deteccion abre la camara equivocada.')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--image', type=str, default=None,
                        help='Procesar una foto fija en vez de la camara.')
    parser.add_argument('--tune', type=str, default=None,
                        choices=list(HSV_RANGES.keys()),
                        help='Lanzar trackbars HSV para calibrar este color.')
    parser.add_argument('--backend', type=str, default='msmf',
                        choices=list(BACKENDS.keys()),
                        help='Backend de captura en Windows. msmf default '
                             '(respeta FOURCC y resolucion mejor). Prueba '
                             'dshow si msmf no abre tu camara.')
    parser.add_argument('--auto-camera', action='store_true',
                        help='No fijar exposicion/wb/foco manuales. Util si '
                             'la iluminacion cambia mucho.')
    parser.add_argument('--list', action='store_true',
                        help='Listar todas las camaras detectables y salir.')
    parser.add_argument('--cell-cm', type=float, default=5.0,
                        help='Tamano de celda del occupancy grid (cm). '
                             'Default 5. Menor = mas resolucion pero A* mas lento.')
    parser.add_argument('--robot-radius', type=float, default=15.0,
                        help='Radio del robot en cm para inflar obstaculos en '
                             'el grid. Default 15. Pon 0 para no inflar.')
    parser.add_argument('--no-grid', action='store_true',
                        help='No construir/mostrar el occupancy grid.')
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    undistort = Undistorter()
    homography = Homography()

    grid_builder = None
    if not args.no_grid and homography.enabled:
        grid_builder = OccupancyGrid(homography,
                                     cell_cm=args.cell_cm,
                                     robot_radius_cm=args.robot_radius)
        print(f"[grid] {grid_builder.rows}x{grid_builder.cols} celdas "
              f"de {args.cell_cm:g} cm (radio robot {args.robot_radius:g} cm)")
    elif args.no_grid:
        print("[grid] desactivado por --no-grid")
    else:
        print("[grid] desactivado: falta homography.json")

    # Resolver indice de camara. Prioridad:
    #  1. Si el usuario paso --camera N explicito -> usar tal cual.
    #  2. Si --camera-name -> auto-detectar via pygrabber + sondeo MSMF.
    #  3. Si nada -> usar indice 0.
    if args.camera is not None:
        # Override explicito del usuario, saltamos auto-deteccion.
        print(f"[camera] --camera={args.camera} explicito, backend="
              f"{args.backend} (saltando auto-deteccion)")
    elif args.camera_name:
        dshow_idx = find_camera_index_by_name(args.camera_name)
        if dshow_idx >= 0:
            msmf_idx = find_msmf_index_for_named_camera(
                dshow_idx, args.width, args.height,
                prefer_other=args.prefer_other_msmf)
            if msmf_idx >= 0:
                args.camera = msmf_idx
                args.backend = 'msmf'
            else:
                args.camera = dshow_idx
                args.backend = 'dshow'
                print("[camera_by_name] Fallback: usando DSHOW (puede bajar "
                      "silenciosamente a 480p).")
        else:
            args.camera = 0
    else:
        args.camera = 0

    print(f"\n** Camara final: index={args.camera}  backend={args.backend} **")
    print(f"   Si abre la camara equivocada, sobrescribe con:")
    print(f"     --camera N --backend msmf   (forzar indice MSMF)")
    print(f"     --prefer-other-msmf          (invertir auto-deteccion)\n")

    # Modo imagen fija
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"No pude leer {args.image}")
        frame = undistort(frame)
        color_dets, obstacles, walls, wall_mask = process_frame(frame)
        attach_world_coords(homography, color_dets, obstacles, walls)
        _print_summary(color_dets, obstacles, walls)
        out = annotate(frame.copy(), color_dets, obstacles, walls)
        cv2.imshow('Detector (imagen)', out)
        if grid_builder is not None:
            occ = grid_builder.build(wall_mask, obstacles)
            grid_img = grid_builder.render(occ, color_dets=color_dets)
            cv2.imshow('Occupancy grid (cm)', grid_img)
            s = grid_builder.stats(occ)
            print(f"  grid: libre={s['pct_free']:.1f}%  "
                  f"pared={s['wall']}  obst={s['obstacle']}  "
                  f"margen={s['inflated']}")
        print("Pulsa cualquier tecla para cerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # Modo tuner
    if args.tune:
        cap = open_camera(args.camera, args.width, args.height, args.backend,
                          manual_exposure=not args.auto_camera)
        try:
            run_tuner(cap, args.tune)
        finally:
            cap.release()
        return

    # Modo live
    cap = open_camera(args.camera, args.width, args.height, args.backend,
                      manual_exposure=not args.auto_camera)
    reader = LatestFrameReader(cap)
    print("Detector cenital corriendo. q=salir, s=snapshot, h=toggle overlay, "
          "g=toggle grid")
    show_overlay = True
    show_grid = grid_builder is not None
    grid_win = 'Occupancy grid (cm)'
    last_log = 0.0
    fps_count = 0
    fps_display = 0.0
    fps_t0 = time.perf_counter()
    try:
        while True:
            ok, frame = reader.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame = undistort(frame)
            color_dets, obstacles, walls, wall_mask = process_frame(frame)
            attach_world_coords(homography, color_dets, obstacles, walls)

            fps_count += 1
            now_perf = time.perf_counter()
            elapsed = now_perf - fps_t0
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_t0 = now_perf

            if show_overlay:
                out = annotate(frame.copy(), color_dets, obstacles, walls)
            else:
                out = frame.copy()
            cv2.putText(out, f"FPS: {fps_display:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow('Detector cenital', out)

            occ = None
            if show_grid and grid_builder is not None:
                occ = grid_builder.build(wall_mask, obstacles)
                grid_img = grid_builder.render(occ, color_dets=color_dets)
                cv2.imshow(grid_win, grid_img)

            now = time.time()
            if now - last_log > 1.0:
                _print_summary(color_dets, obstacles, walls, compact=True)
                if occ is not None:
                    s = grid_builder.stats(occ)
                    print(f"  grid: libre={s['pct_free']:.1f}%  "
                          f"pared={s['wall']}  obst={s['obstacle']}  "
                          f"margen={s['inflated']}")
                last_log = now

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('h'):
                show_overlay = not show_overlay
            if k == ord('g') and grid_builder is not None:
                show_grid = not show_grid
                if not show_grid:
                    cv2.destroyWindow(grid_win)
            if k == ord('s'):
                fn = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fn, frame)
                print(f"Guardado {fn}")
    finally:
        reader.stop()
        cap.release()
        cv2.destroyAllWindows()


def _print_summary(color_dets, obstacles, walls=None, compact=False):
    counts = {c: {'CUBE': 0, 'ZONE': 0, 'UNKNOWN': 0}
              for c in color_dets}
    for c, dets in color_dets.items():
        for d in dets:
            counts[c][d['kind']] += 1
    n_walls = len(walls) if walls else 0
    if compact:
        parts = []
        for c, k in counts.items():
            parts.append(f"{c}:C{k['CUBE']}/Z{k['ZONE']}")
        parts.append(f"OBST:{len(obstacles)}")
        parts.append(f"WALL:{n_walls}")
        print(" | ".join(parts))
    else:
        for c, k in counts.items():
            print(f"  {c}: cubos={k['CUBE']}  zonas={k['ZONE']}  "
                  f"unknown={k['UNKNOWN']}")
        print(f"  obstaculos: {len(obstacles)}")
        print(f"  paredes:    {n_walls}")


if __name__ == '__main__':
    main()
