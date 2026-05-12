"""
calibrate_camera.py - Calibracion intrinseca de la camara cenital.

Calcula la matriz K (focal + centro) y los coeficientes de distorsion para
poder corregir el "ojo de pescado" de la C920. Guarda el resultado en
camera_params.json. El detector lo carga automaticamente si esta presente.

------------------------------------------------------------------------------
PREPARACION (una sola vez):
------------------------------------------------------------------------------
1. Imprime un tablero de ajedrez. Recomendacion: 9x6 esquinas internas,
   30 mm por casilla. Hay PDF gratis aqui:
     https://github.com/opencv/opencv/blob/4.x/doc/pattern.png
   Pegalo SOBRE algo PLANO (carton rigido, no doblado).

2. Mide el lado real de una casilla con regla y ajusta SQUARE_SIZE_MM abajo.

------------------------------------------------------------------------------
USO:
------------------------------------------------------------------------------
    # Modo captura: mueve el tablero, pulsa SPACE para capturar (15-25 fotos)
    python calibrate_camera.py --camera 1 --capture

    # Modo calcular: procesa las fotos en calib_images/ y guarda los parametros
    python calibrate_camera.py --solve

    # Modo verificar: abre la camara con undistort aplicado para ver el efecto
    python calibrate_camera.py --camera 1 --verify

CONSEJOS PARA QUE LA CALIBRACION SALGA BIEN:
  - Mueve el tablero a TODAS las zonas de la imagen (esquinas, centro, bordes)
  - Inclinalo en diferentes angulos (de frente, de lado, rotado)
  - Manten el tablero plano y QUIETO al capturar
  - Buena iluminacion, sin reflejos brillantes
  - 15-25 fotos validas son suficientes; mas fotos = mas precision pero
    rendimiento decreciente
  - El error medio (RMS) debe quedar < 0.5 px para una buena calibracion
"""

import argparse
import glob
import json
import os
import time

os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2  # noqa: E402
import numpy as np  # noqa: E402


# ---- AJUSTA ESTOS A TU TABLERO ---------------------------------------------
PATTERN_SIZE   = (9, 6)        # esquinas INTERNAS (no casillas)
SQUARE_SIZE_MM = 23.0          # lado real de una casilla en milimetros
# -----------------------------------------------------------------------------

IMAGES_DIR     = os.path.join(os.path.dirname(__file__), 'calib_images')
OUTPUT_FILE    = os.path.join(os.path.dirname(__file__), 'camera_params.json')


def _open_camera(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir camara {index}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(10):
        cap.read()
        time.sleep(0.05)
    return cap


def capture_mode(camera, width, height):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    cap = _open_camera(camera, width, height)
    print(f"Modo CAPTURA. Pulsa SPACE para guardar, ESC/q para salir.")
    print(f"Guardando en: {IMAGES_DIR}")

    n = len(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, PATTERN_SIZE, corners, found)
            cv2.putText(vis, "TABLERO OK - pulsa SPACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(vis, "buscando tablero...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis, f"Capturadas: {n}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Calibracion - captura', vis)

        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord('q')):
            break
        if k == ord(' ') and found:
            fn = os.path.join(IMAGES_DIR, f"calib_{n:03d}.png")
            cv2.imwrite(fn, frame)
            n += 1
            print(f"  guardada {fn}  (total {n})")
    cap.release()
    cv2.destroyAllWindows()
    print(f"Listo. {n} imagenes capturadas. Ahora corre con --solve.")


def solve_mode():
    files = sorted(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
    if len(files) < 8:
        raise SystemExit(
            f"Solo {len(files)} imagenes en {IMAGES_DIR}. Necesitas >= 8 "
            f"(idealmente 15-25). Corre con --capture primero.")

    # Puntos 3D del tablero en su propio sistema (z=0)
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0],
                           0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    obj_points = []
    img_points = []
    img_size = None
    used = 0
    rejected = 0

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            print(f"  RECHAZADA (no tablero): {os.path.basename(f)}")
            rejected += 1
            continue
        # Refinar a sub-pixel
        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        used += 1
        print(f"  OK: {os.path.basename(f)}")

    if used < 8:
        raise SystemExit(f"Solo {used} imagenes validas. Necesitas >= 8.")

    print(f"\nResolviendo con {used} imagenes (rechazadas {rejected})...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    print(f"\nRMS reproyeccion: {rms:.4f} px  "
          f"({'EXCELENTE' if rms < 0.3 else 'BIEN' if rms < 0.5 else 'mejorable, considera tomar mas fotos / variar mas el angulo'})")
    print(f"Resolucion: {img_size[0]}x{img_size[1]}")
    print(f"\nMatriz K:\n{K}")
    print(f"\nCoeficientes de distorsion (k1, k2, p1, p2, k3): {dist.ravel()}")

    data = {
        'image_width':  img_size[0],
        'image_height': img_size[1],
        'rms':          float(rms),
        'camera_matrix': K.tolist(),
        'distortion_coeffs': dist.ravel().tolist(),
        'square_size_mm': SQUARE_SIZE_MM,
        'pattern_size': list(PATTERN_SIZE),
        'num_images_used': used,
    }
    with open(OUTPUT_FILE, 'w') as fh:
        json.dump(data, fh, indent=2)
    print(f"\nGuardado en {OUTPUT_FILE}")


def verify_mode(camera, width, height):
    if not os.path.exists(OUTPUT_FILE):
        raise SystemExit(f"No existe {OUTPUT_FILE}. Corre --solve primero.")
    with open(OUTPUT_FILE) as fh:
        data = json.load(fh)
    K = np.array(data['camera_matrix'])
    dist = np.array(data['distortion_coeffs'])

    cap = _open_camera(camera, width, height)
    print("VERIFICACION: arriba=original, abajo=corregido. q=salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        undist = cv2.undistort(frame, K, dist, None, new_K)
        comp = np.vstack([frame, undist])
        cv2.imshow('Original (arriba) vs Corregido (abajo)',
                   cv2.resize(comp, (w, h)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument('--capture', action='store_true',
                     help='Capturar imagenes del tablero.')
    grp.add_argument('--solve', action='store_true',
                     help='Procesar imagenes capturadas y calcular K + dist.')
    grp.add_argument('--verify', action='store_true',
                     help='Mostrar original vs corregido en vivo.')
    args = parser.parse_args()

    if args.capture:
        capture_mode(args.camera, args.width, args.height)
    elif args.solve:
        solve_mode()
    elif args.verify:
        verify_mode(args.camera, args.width, args.height)


if __name__ == '__main__':
    main()
