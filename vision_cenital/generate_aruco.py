"""
generate_aruco.py - Genera un marcador ArUco listo para imprimir.

Crea un PNG con el marcador centrado y un borde blanco (quiet zone) que
ArUco necesita para detectarlo de forma robusta. El PNG queda dimensionado
para imprimirse exactamente al tamano fisico que pidas.

Uso tipico:
    # Marcador ID 0, 10 cm de lado, a 300 DPI -> aruco_id0_10cm.png
    python generate_aruco.py

    # Otro tamano y otro ID
    python generate_aruco.py --id 1 --size-cm 8

Diccionario usado: DICT_4X4_50
  - 4x4 bloques internos = bloques grandes = detectable desde lejos
  - 50 IDs disponibles (mas que suficiente: 1 marcador para el robot)

Como imprimir bien:
  1. Abre el PNG generado
  2. Imprime a "tamano real" / "100%" / sin ajustar a pagina
  3. Al imprimir, asegurate que el cuadro negro mida EXACTAMENTE size_cm
     con una regla. Si imprime mas pequeno por ajustes del driver, pegalo
     y dime el tamano REAL para configurar el detector.

Como pegarlo al robot:
  - Lo mas plano posible (carton rigido o cartulina pegada arriba)
  - Una orientacion clara: la flecha en la esquina superior izquierda del
    marcador marca el "frente" del robot. Decide tu orientacion y se
    consistente.
  - Con buena luz, no en una zona con sombras o reflejos brillantes
"""

import argparse
import os

import cv2
import numpy as np


# Hay dos APIs segun version de opencv-python. Detectamos cual existe.
def _get_dict(dict_name):
    aruco_mod = cv2.aruco
    dict_id = getattr(aruco_mod, dict_name)
    # API nueva (opencv >= 4.7)
    if hasattr(aruco_mod, 'getPredefinedDictionary'):
        return aruco_mod.getPredefinedDictionary(dict_id)
    # API vieja (opencv < 4.7)
    return aruco_mod.Dictionary_get(dict_id)


def _draw_marker(aruco_dict, marker_id, side_px):
    aruco_mod = cv2.aruco
    if hasattr(aruco_mod, 'generateImageMarker'):
        return aruco_mod.generateImageMarker(aruco_dict, marker_id, side_px)
    return aruco_mod.drawMarker(aruco_dict, marker_id, side_px)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0,
                        help='ID del marcador (0..49). Default 0.')
    parser.add_argument('--size-cm', type=float, default=10.0,
                        help='Tamano fisico del cuadrado negro en cm. '
                             'Default 10. Ajusta segun el techo del robot.')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Resolucion de impresion (puntos por pulgada). '
                             'Default 300. Bajalo a 150 si la impresora es '
                             'limitada.')
    parser.add_argument('--quiet-zone-cm', type=float, default=1.5,
                        help='Borde blanco alrededor del marcador en cm. '
                             'Necesario para deteccion fiable. Default 1.5.')
    parser.add_argument('--dict', type=str, default='DICT_4X4_50',
                        help='Diccionario ArUco. Default DICT_4X4_50.')
    parser.add_argument('--out', type=str, default=None,
                        help='Archivo de salida. Por defecto '
                             'aruco_id<ID>_<SIZE>cm.png en este directorio.')
    parser.add_argument('--with-arrow', action='store_true', default=True,
                        help='Anadir una flecha y el ID alrededor del '
                             'marcador como guia para pegarlo orientado.')
    args = parser.parse_args()

    aruco_dict = _get_dict(args.dict)

    # cm -> pixeles
    cm_per_inch = 2.54
    px_per_cm = args.dpi / cm_per_inch
    marker_px = int(round(args.size_cm * px_per_cm))
    qz_px = int(round(args.quiet_zone_cm * px_per_cm))

    # Generar el marcador en su tamano final (sin re-escalar para no
    # introducir blur en los bordes)
    marker = _draw_marker(aruco_dict, args.id, marker_px)

    # Lienzo blanco con el marcador centrado
    total = marker_px + 2 * qz_px
    canvas = np.full((total, total), 255, dtype=np.uint8)
    canvas[qz_px:qz_px + marker_px, qz_px:qz_px + marker_px] = marker

    # Anadir info alrededor del marcador para que no nos confundamos al
    # pegarlo: flecha indicando "frente del robot" y el ID
    if args.with_arrow:
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        # Texto ID arriba
        cv2.putText(canvas_bgr, f"ID={args.id}",
                    (qz_px, qz_px // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2,
                    cv2.LINE_AA)
        # Texto tamano abajo a la izquierda
        cv2.putText(canvas_bgr, f"{args.size_cm:.1f} cm",
                    (qz_px, total - qz_px // 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2,
                    cv2.LINE_AA)
        # Flecha "FRENTE" apuntando al lado izquierdo del marcador (es el
        # lado X+ en el sistema interno de ArUco)
        ay = qz_px + marker_px // 2
        ax1 = qz_px // 2 - 10
        ax2 = qz_px // 2 + 20
        cv2.arrowedLine(canvas_bgr, (ax1, ay), (ax2, ay),
                        (80, 80, 80), 2, tipLength=0.4)
        cv2.putText(canvas_bgr, "FRENTE",
                    (5, ay - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1,
                    cv2.LINE_AA)
        canvas = canvas_bgr

    # Nombre de salida
    out_path = args.out
    if out_path is None:
        out_path = os.path.join(
            os.path.dirname(__file__),
            f"aruco_id{args.id}_{args.size_cm:.0f}cm.png")

    cv2.imwrite(out_path, canvas)

    print(f"Generado: {out_path}")
    print(f"  Marcador:      {args.size_cm:.1f} cm x {args.size_cm:.1f} cm "
          f"({marker_px} px)")
    print(f"  Quiet zone:    {args.quiet_zone_cm:.1f} cm a cada lado "
          f"({qz_px} px)")
    print(f"  Tamano total:  {total} x {total} px @ {args.dpi} DPI")
    print(f"  Diccionario:   {args.dict}, ID {args.id}")
    print()
    print("Imprime a TAMANO REAL (sin 'ajustar a pagina'). Despues, mide")
    print("con regla el lado del cuadro negro; deberia dar "
          f"{args.size_cm:.1f} cm. Si no, dime el tamano real medido.")


if __name__ == '__main__':
    main()
