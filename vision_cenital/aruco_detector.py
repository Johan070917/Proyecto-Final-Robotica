"""
aruco_detector.py - Detector de marcadores ArUco para localizar al robot.

Lee un frame, encuentra el marcador del robot y devuelve su pose: posicion
y orientacion. Si tiene homografia disponible, devuelve la pose en
coordenadas mundo (cm); si no, en pixeles.

Convencion de orientacion:
  - El angulo se mide en el plano del campo (espacio cm), no en pixeles.
  - 0 grados = el eje X+ del marcador apunta hacia X+ del mundo (a la
    derecha en la imagen).
  - Crece en sentido horario porque el eje Y mundo apunta hacia ABAJO
    (igual convencion que la homografia).

Uso tipico:

    aruco = RobotDetector(homography, marker_id=0, marker_size_cm=10.0)
    pose  = aruco.detect(frame)
    if pose is not None:
        print(pose['pos_cm'], pose['angle_deg'])
        aruco.draw_on_frame(frame, pose)
"""

import math

import cv2
import numpy as np


# Una sola fuente de verdad: estos defaults deben coincidir con los flags
# CLI del detector.
DEFAULT_DICT          = 'DICT_4X4_50'
DEFAULT_MARKER_ID     = 0
DEFAULT_MARKER_CM     = 10.0


def _tune_params(params):
    """Ajusta los parametros para deteccion robusta de marcadores pequenos
    (~80 px de lado). Los defaults de OpenCV asumen marcadores grandes.
    """
    # Refinamiento sub-pixel de las esquinas: mejor pose y mas estable
    # frame a frame.
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy = 0.05

    # Umbralizado adaptativo: ventanas mas pequenas para no perder marcadores
    # que ocupan pocas celdas en pantalla. Si la luz cambia entre frames el
    # default se queda corto.
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant = 7

    # Permitir detectar marcadores pequenos: bajar el perimetro minimo del
    # 3% al 1.5% del lado de la imagen. Con cenital a 1280x720, 1.5% son
    # ~11 px de perimetro, mas que suficiente.
    params.minMarkerPerimeterRate = 0.015
    params.maxMarkerPerimeterRate = 4.0

    # Aceptar contornos un poco menos cuadrados (compensar borrosidad de
    # movimiento). 0.05 default; subir a 0.08 ayuda con motion blur leve.
    params.polygonalApproxAccuracyRate = 0.08

    # Mas permisivo con la separacion entre el marcador y su quiet zone
    # (por si el marcador esta pegado cerca de algo oscuro).
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 1


def _get_dict_and_detector(dict_name):
    """Devuelve (aruco_dict, detector_or_None, params). detector_or_None es
    el objeto ArucoDetector si hay API nueva; params se devuelve siempre por
    si hace falta usarlo con la API funcional legacy.
    """
    aruco_mod = cv2.aruco
    dict_id = getattr(aruco_mod, dict_name)
    if hasattr(aruco_mod, 'getPredefinedDictionary'):
        # API >= 4.7
        d = aruco_mod.getPredefinedDictionary(dict_id)
        params = aruco_mod.DetectorParameters()
        _tune_params(params)
        detector = aruco_mod.ArucoDetector(d, params)
        return d, detector, params
    # API < 4.7
    d = aruco_mod.Dictionary_get(dict_id)
    params = aruco_mod.DetectorParameters_create()
    _tune_params(params)
    return d, None, params


class RobotDetector:
    """Detecta UN marcador ArUco (el del robot) y devuelve su pose."""

    def __init__(self, homography=None, dict_name=DEFAULT_DICT,
                 marker_id=DEFAULT_MARKER_ID,
                 marker_size_cm=DEFAULT_MARKER_CM, use_clahe=True):
        self.homography = homography
        self.marker_id = marker_id
        self.marker_size_cm = marker_size_cm
        self._dict, self._detector, self._params_legacy = \
            _get_dict_and_detector(dict_name)
        # CLAHE = ecualizacion adaptativa con limite. Iguala contraste local
        # sin amplificar ruido (a diferencia de la ecualizacion global). En
        # imagenes con sombras y reflejos mejora notablemente la deteccion.
        self._clahe = (cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                       if use_clahe else None)

    # ----------------------------------------------------------- detection
    def _try_detect(self, gray):
        """Una pasada de detectMarkers. Devuelve (corners_list, ids)."""
        if self._detector is not None:
            corners_list, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners_list, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dict, parameters=self._params_legacy)
        return corners_list, ids

    def detect(self, frame):
        """Detecta el marcador. Devuelve None si no aparece.

        En caso afirmativo devuelve un dict con:
          - 'id'          : int
          - 'corners_px'  : numpy (4,2) con las 4 esquinas en pixeles
                            (orden ArUco: TL, TR, BR, BL del marcador)
          - 'center_px'   : (cx, cy) en pixeles
          - 'front_px'    : punto medio del lado TL-TR en pixeles (frente
                            del marcador, eje X+)
          - 'angle_deg'   : orientacion en grados (en el espacio mundo si
                            hay homografia; si no, en imagen)
          - 'pos_cm'      : (x, y) en cm si hay homografia; si no None
          - 'front_cm'    : punto frente en cm; None sin homografia
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Intentamos primero sobre la imagen original. Si no encuentra nada,
        # probamos con CLAHE aplicado (mas costoso, pero rescata frames con
        # baja luz / sombras). Asi en condiciones normales no pagamos coste.
        corners_list, ids = self._try_detect(gray)
        if (ids is None or len(ids) == 0) and self._clahe is not None:
            gray_eq = self._clahe.apply(gray)
            corners_list, ids = self._try_detect(gray_eq)

        if ids is None or len(ids) == 0:
            return None

        # Buscar el marcador del robot por ID
        target_idx = None
        for i, mid in enumerate(ids.flatten()):
            if int(mid) == self.marker_id:
                target_idx = i
                break
        if target_idx is None:
            return None

        c = corners_list[target_idx][0].astype(np.float32)  # (4,2)
        center_px = tuple(c.mean(axis=0).tolist())
        # Punto medio del lado IZQUIERDO del marcador (TL -> BL).
        # Esto coincide con la flecha "FRENTE" del PNG que genera
        # generate_aruco.py: apunta saliendo por el lado izquierdo del
        # cuadrado negro. El usuario pega el marcador con esa flecha
        # mirando al frente del robot, asi que el frente del robot es
        # este punto.
        front_px = tuple(((c[0] + c[3]) / 2.0).tolist())

        pos_cm = None
        front_cm = None
        if self.homography is not None and self.homography.enabled:
            both = self.homography.to_cm([center_px, front_px])
            if both is not None:
                pos_cm, front_cm = both

        # Angulo. Lo medimos en el espacio donde lo vamos a usar (cm si hay
        # homografia, pixeles si no). Asi es coherente con el grid.
        if pos_cm is not None and front_cm is not None:
            dx = front_cm[0] - pos_cm[0]
            dy = front_cm[1] - pos_cm[1]
        else:
            dx = front_px[0] - center_px[0]
            dy = front_px[1] - center_px[1]
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)

        return {
            'id':         int(self.marker_id),
            'corners_px': c,
            'center_px':  center_px,
            'front_px':   front_px,
            'angle_deg':  angle_deg,
            'angle_rad':  angle_rad,
            'pos_cm':     pos_cm,
            'front_cm':   front_cm,
        }

    # --------------------------------------------------------------- drawing
    def draw_on_frame(self, frame, pose):
        """Dibuja contorno del marcador + flecha del frente sobre el frame."""
        if pose is None:
            return frame
        c = pose['corners_px'].astype(int)
        cv2.polylines(frame, [c.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
        # Esquina TL marcada en verde como referencia
        cv2.circle(frame, tuple(c[0]), 5, (0, 255, 0), -1)
        # Flecha del centro al frente
        ctr = tuple(int(v) for v in pose['center_px'])
        fr = tuple(int(v) for v in pose['front_px'])
        cv2.arrowedLine(frame, ctr, fr, (0, 255, 255), 2, tipLength=0.3)
        label = f"ROBOT id={pose['id']}"
        if pose['pos_cm'] is not None:
            x, y = pose['pos_cm']
            label += f"  ({x:.0f},{y:.0f})cm  {pose['angle_deg']:+.0f}deg"
        cv2.putText(frame, label, (c[0][0], max(0, c[0][1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2,
                    cv2.LINE_AA)
        return frame

    def draw_on_grid(self, grid_img, pose, grid_builder):
        """Dibuja el robot en la imagen del occupancy grid.

        grid_builder: instancia de OccupancyGrid (para conversion cm -> px).
        """
        if pose is None or pose['pos_cm'] is None:
            return grid_img
        x_cm, y_cm = pose['pos_cm']
        scale = grid_img.shape[0] / grid_builder.rows  # px por celda
        # cm -> px de la imagen del grid
        px = int(x_cm / grid_builder.cell_cm * scale)
        py = int(y_cm / grid_builder.cell_cm * scale)

        # Dibujar el robot como rectangulo orientado
        # (largo paralelo al frente, ancho perpendicular)
        robot_len = 20.0   # cm
        robot_wid = 19.5   # cm
        half_l = robot_len / 2.0 / grid_builder.cell_cm * scale
        half_w = robot_wid / 2.0 / grid_builder.cell_cm * scale
        ang = pose['angle_rad']
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        # 4 esquinas del rectangulo en marco local, luego rotadas y trasladadas
        local = [(+half_l, -half_w), (+half_l, +half_w),
                 (-half_l, +half_w), (-half_l, -half_w)]
        pts = []
        for lx, ly in local:
            gx = lx * cos_a - ly * sin_a + px
            gy = lx * sin_a + ly * cos_a + py
            pts.append((int(gx), int(gy)))
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(grid_img, [pts_np], (0, 200, 255))
        cv2.polylines(grid_img, [pts_np], True, (0, 0, 0), 1)

        # Flecha indicando el frente
        front_x = int(px + half_l * cos_a)
        front_y = int(py + half_l * sin_a)
        cv2.arrowedLine(grid_img, (px, py), (front_x, front_y),
                        (0, 0, 0), 2, tipLength=0.5)
        return grid_img
