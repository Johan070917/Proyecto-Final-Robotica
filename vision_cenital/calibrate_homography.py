"""
calibrate_homography.py - Mapeo pixel -> coordenadas reales en cm.

Captura un frame de la camara cenital, te deja hacer clic en las 4 esquinas
del campo, y calcula la matriz de homografia H que convierte cualquier punto
(px, py) en (x_cm, y_cm) en el plano del suelo.

Sistema de coordenadas (cm):
  - Origen (0, 0) en la esquina SUPERIOR IZQUIERDA del campo
  - X positivo hacia la DERECHA  (eje largo, 404.5 cm)
  - Y positivo hacia ABAJO       (eje ancho,  210 cm)

Uso:
    python calibrate_homography.py --prefer-other-msmf

Pasos:
    1. Se abre la camara, pulsa SPACE para congelar el frame
    2. Haces clic en las 4 esquinas en este ORDEN:
       1) Superior IZQUIERDA   -> (0,     0)   cm
       2) Superior DERECHA     -> (404.5, 0)   cm
       3) Inferior DERECHA     -> (404.5, 210) cm
       4) Inferior IZQUIERDA   -> (0,     210) cm
    3. Pulsa SPACE para calcular H. Se abre una ventana extra mostrando la
       vista rectificada (campo visto desde arriba en cm).
    4. Pulsa cualquier tecla para guardar y salir. Salida: homography.json

Teclas:
    SPACE  capturar frame / aceptar clicks / cerrar verificacion
    r      reiniciar clicks
    q      salir sin guardar
"""

import argparse
import json
import os
import time

os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from detector import (
    Undistorter,
    find_camera_index_by_name,
    find_msmf_index_for_named_camera,
    open_camera,
    HAS_PYGRABBER,
)


# --- AJUSTA SI TU CAMPO CAMBIA ---------------------------------------------
FIELD_W_CM = 404.5   # largo total (mitad izq 206.5 + mitad der 198)
FIELD_H_CM = 210.0   # ancho total
# ---------------------------------------------------------------------------

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'homography.json')

CORNER_LABELS = [
    "1) Superior IZQUIERDA",
    "2) Superior DERECHA",
    "3) Inferior DERECHA",
    "4) Inferior IZQUIERDA",
]

CORNERS_CM = np.array([
    [0.0,        0.0       ],
    [FIELD_W_CM, 0.0       ],
    [FIELD_W_CM, FIELD_H_CM],
    [0.0,        FIELD_H_CM],
], dtype=np.float32)


def _draw_state(frame, clicks):
    vis = frame.copy()
    h_img = vis.shape[0]

    # Hint arriba
    if len(clicks) < 4:
        msg = f"Haz CLIC en: {CORNER_LABELS[len(clicks)]}   "
        msg += "(r=reset, q=salir)"
        color = (0, 255, 255)
    else:
        msg = "4 clicks listos. Pulsa SPACE para calcular  (r=reset)"
        color = (0, 255, 0)
    cv2.putText(vis, msg, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    # Puntos
    for i, (x, y) in enumerate(clicks):
        cv2.circle(vis, (x, y), 10, (0, 255, 0), 2)
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(vis, str(i + 1), (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    cv2.LINE_AA)

    # Lineas entre puntos
    n = len(clicks)
    for i in range(n - 1):
        cv2.line(vis, clicks[i], clicks[i + 1], (0, 200, 0), 2)
    if n == 4:
        cv2.line(vis, clicks[3], clicks[0], (0, 200, 0), 2)

    return vis


def _capture_stable_frame(cap, undistort):
    """Muestra preview, captura frame al pulsar SPACE."""
    print("\nMueve la camara si hace falta. Pulsa SPACE para CONGELAR el frame,")
    print("q para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        if undistort is not None:
            frame = undistort(frame)
        vis = frame.copy()
        cv2.putText(vis, "SPACE = congelar frame   q = salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.imshow("Calibracion Homografia - preview", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow("Calibracion Homografia - preview")
            return None
        if k == ord(' '):
            cv2.destroyWindow("Calibracion Homografia - preview")
            return frame


def _click_corners(frame):
    """Devuelve lista de 4 (x, y) clicks o None si el usuario cancela."""
    clicks = []
    win = "Calibracion Homografia - clic en las 4 esquinas"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))
            print(f"  {CORNER_LABELS[len(clicks) - 1]} -> ({x}, {y})")

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    print("\nHaz clic en las 4 esquinas del campo EN ESTE ORDEN:")
    for label in CORNER_LABELS:
        print(f"  {label}")

    while True:
        cv2.imshow(win, _draw_state(frame, clicks))
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            cv2.destroyWindow(win)
            return None
        if k == ord('r'):
            clicks.clear()
            print("  Reset.")
        if k == ord(' ') and len(clicks) == 4:
            cv2.destroyWindow(win)
            return clicks


def _verify_homography(frame, H):
    """Muestra el campo rectificado: 1 px = 1 cm aprox."""
    out_w = int(FIELD_W_CM)
    out_h = int(FIELD_H_CM)
    warped = cv2.warpPerspective(frame, H, (out_w, out_h))
    win = "Vista rectificada (1 px ~ 1 cm).   Tecla cualquiera = guardar"
    cv2.imshow(win, warped)
    print("\nVerifica la vista rectificada. Cualquier tecla = guardar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=None)
    parser.add_argument('--camera-name', type=str, default='C920')
    parser.add_argument('--prefer-other-msmf', action='store_true')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--backend', type=str, default='msmf')
    args = parser.parse_args()

    # --- Resolver indice de camara (igual logica que detector.py) ----------
    if args.camera is None and HAS_PYGRABBER:
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
        else:
            args.camera = 0
    elif args.camera is None:
        args.camera = 0

    print(f"\n** Camara: index={args.camera}  backend={args.backend} **\n")

    cap = open_camera(args.camera, args.width, args.height, args.backend)
    undistort = Undistorter()
    if not undistort.enabled:
        print("\n[!] AVISO: no encontre camera_params.json. La homografia se")
        print("    calculara sobre la imagen DISTORSIONADA, lo que da error en")
        print("    los bordes. Calibra la camara intrinsicamente primero con")
        print("    calibrate_camera.py para mejor precision.\n")
        try:
            input("Pulsa ENTER para continuar igualmente, o Ctrl+C para abortar... ")
        except KeyboardInterrupt:
            cap.release()
            return

    # --- Capturar frame ----------------------------------------------------
    frame = _capture_stable_frame(cap, undistort if undistort.enabled else None)
    cap.release()
    if frame is None:
        print("Cancelado.")
        return

    # --- Clicks ------------------------------------------------------------
    clicks = _click_corners(frame)
    if clicks is None:
        print("Cancelado.")
        return

    # --- Calcular H --------------------------------------------------------
    src = np.array(clicks, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, CORNERS_CM)

    print(f"\nMatriz H (pixel -> cm):\n{H}\n")

    # Sanity check: la esquina 1 deberia mapear a (0,0)
    test = cv2.perspectiveTransform(
        np.array([[clicks[0]]], dtype=np.float32), H)[0, 0]
    print(f"Test: esquina 1 ({clicks[0]}) mapea a "
          f"({test[0]:.2f}, {test[1]:.2f}) cm  -- deberia ser (0, 0)")

    # --- Verificar visualmente ---------------------------------------------
    _verify_homography(frame, H)

    # --- Guardar -----------------------------------------------------------
    data = {
        'image_corners_px':  [list(c) for c in clicks],
        'world_corners_cm':  CORNERS_CM.tolist(),
        'field_width_cm':    FIELD_W_CM,
        'field_height_cm':   FIELD_H_CM,
        'homography':        H.tolist(),
        'undistorted':       bool(undistort.enabled),
        'image_size':        [frame.shape[1], frame.shape[0]],
    }
    with open(OUTPUT_FILE, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f"\nGuardado en {OUTPUT_FILE}")
    print("\nProximo paso: integrar la homografia al detector para que muestre")
    print("posiciones en cm junto a cada deteccion.")


if __name__ == '__main__':
    main()
