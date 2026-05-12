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
"""

import argparse
import json
import os
import time

# Desactiva las "HW transforms" de MSMF antes de importar cv2. Sin esto,
# probar un indice de camara inexistente con MSMF en Windows cuelga sin
# timeout. Tiene que estar ANTES de "import cv2".
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    from pygrabber.dshow_graph import FilterGraph
    HAS_PYGRABBER = True
except ImportError:
    HAS_PYGRABBER = False


CAMERA_PARAMS_FILE = os.path.join(os.path.dirname(__file__),
                                  'camera_params.json')


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
            self.new_K, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.dist, (w, h), 1, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2)
            self._size = (w, h)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


# -----------------------------------------------------------------------------
# Rangos HSV iniciales (calibrar con --tune)
# -----------------------------------------------------------------------------
HSV_RANGES = {
    'RED':   [((0,   110,  80), (10,  255, 255)),
              ((170, 110,  80), (180, 255, 255))],
    'GREEN': [((34,   26,  40), (86,  255, 255))],
    'BLUE':  [((95,  110,  60), (130, 255, 255))],
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

# Umbral de fill-ratio (raw_mask_count / bbox_area) para clasificar cubo vs zona
FILL_CUBE_MIN  = 0.65      # > este = cubo
FILL_ZONE_MAX  = 0.55      # < este = zona
# Entre 0.55 y 0.65 = UNKNOWN

# Kernel para "rellenar" la zona con X. Debe escalar con el tamano del blob,
# asi que lo aplicamos relativo y no absoluto (ver detect_color_objects).
CLOSE_KERNEL_REL = 0.30    # 30% del lado del bbox aprox

# Filtros para descartar la cinta-linea del puente (largo y muy delgado).
# Subimos el limite para que paredes (rectangulos largos pero gruesos) si
# pasen el filtro. La cinta del puente tiene aspect > 20.
OBST_ASPECT_MAX = 12.0
OBST_EXTENT_MIN = 0.35


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
        fill = cv2.countNonZero(roi) / float(w * h)

        if fill >= FILL_CUBE_MIN:
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


def detect_obstacles(hsv):
    """Detecta blobs negros. Filtra cinta-linea (larga y delgada)."""
    lo, hi = HSV_BLACK
    mask = cv2.inRange(hsv, np.array(lo, dtype=np.uint8),
                       np.array(hi, dtype=np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA_OBST:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w == 0 or h == 0:
            continue
        aspect = max(w / h, h / w)
        if aspect > OBST_ASPECT_MAX:
            continue           # cinta-linea
        extent = area / float(w * h)
        if extent < OBST_EXTENT_MIN:
            continue           # forma muy irregular
        out.append({
            'bbox': (x, y, w, h),
            'center': (x + w // 2, y + h // 2),
            'area': area,
        })
    return out


# -----------------------------------------------------------------------------
def annotate(frame, color_dets, obstacles):
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

    for o in obstacles:
        x, y, w, h = o['bbox']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 60, 60), 2)
        cv2.putText(frame, "OBST", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 2)
    return frame


def process_frame(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_dets = {
        c: detect_color_objects(hsv, c, r) for c, r in HSV_RANGES.items()
    }
    obstacles = detect_obstacles(hsv)
    return color_dets, obstacles


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
    """Busca el indice de una camara cuyo nombre contenga la subcadena dada.

    Usa DirectShow via pygrabber. Devuelve -1 si no encuentra coincidencia.
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
                  f"'{n}' en indice {i}")
            return i
    print(f"[camera_by_name] No encontre camara con '{name_substring}'. "
          f"Camaras detectadas: {names}")
    return -1


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
    parser.add_argument('--camera', type=int, default=0,
                        help='Indice de la camara (0,1,2...). Default 0. '
                             'Ignorado si --camera-name encuentra la camara.')
    parser.add_argument('--camera-name', type=str, default='C920',
                        help='Buscar la camara por subcadena del nombre '
                             '(default "C920"). Mas robusto que --camera '
                             'porque el indice cambia entre reconexiones. '
                             'Si no encuentra, cae a --camera.')
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
    args = parser.parse_args()

    if args.list:
        list_cameras()
        return

    undistort = Undistorter()

    # Resolver indice de camara: primero por nombre, si falla usa --camera
    if args.camera_name:
        found = find_camera_index_by_name(args.camera_name)
        if found >= 0:
            args.camera = found

    # Modo imagen fija
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"No pude leer {args.image}")
        frame = undistort(frame)
        color_dets, obstacles = process_frame(frame)
        _print_summary(color_dets, obstacles)
        out = annotate(frame.copy(), color_dets, obstacles)
        cv2.imshow('Detector (imagen)', out)
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
    print("Detector cenital corriendo. q=salir, s=snapshot, h=toggle overlay")
    show_overlay = True
    last_log = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = undistort(frame)
            color_dets, obstacles = process_frame(frame)

            if show_overlay:
                out = annotate(frame.copy(), color_dets, obstacles)
            else:
                out = frame
            cv2.imshow('Detector cenital', out)

            now = time.time()
            if now - last_log > 1.0:
                _print_summary(color_dets, obstacles, compact=True)
                last_log = now

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('h'):
                show_overlay = not show_overlay
            if k == ord('s'):
                fn = f"snapshot_{int(time.time())}.png"
                cv2.imwrite(fn, frame)
                print(f"Guardado {fn}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _print_summary(color_dets, obstacles, compact=False):
    counts = {c: {'CUBE': 0, 'ZONE': 0, 'UNKNOWN': 0}
              for c in color_dets}
    for c, dets in color_dets.items():
        for d in dets:
            counts[c][d['kind']] += 1
    if compact:
        parts = []
        for c, k in counts.items():
            parts.append(f"{c}:C{k['CUBE']}/Z{k['ZONE']}")
        parts.append(f"OBST:{len(obstacles)}")
        print(" | ".join(parts))
    else:
        for c, k in counts.items():
            print(f"  {c}: cubos={k['CUBE']}  zonas={k['ZONE']}  "
                  f"unknown={k['UNKNOWN']}")
        print(f"  obstaculos: {len(obstacles)}")


if __name__ == '__main__':
    main()
