"""
vision_node.py  -  Deteccion de cubos de color con la Camera Module 3.

Salida ROS:
  * /cube_detected   (std_msgs/String)   "COLOR,cx,cy,wpx,dist_m,ang_rad"
  * /vision/image    (sensor_msgs/Image) imagen anotada (para rqt_image_view)

Visualizacion local:
  El nodo abre una ventana OpenCV ("cv2.imshow") con los rectangulos
  dibujados sobre los cubos detectados.  Para verla desde el PC, conecta
  por SSH con X11 forwarding:

      ssh -X johan@<IP_RPI>
      ros2 run diff_bot_3 vision_node

  La ventana se abrira en tu PC siempre que tengas un servidor X
  (VcXsrv en Windows, X11 nativo en Linux).
"""

import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from cv_bridge import CvBridge
    HAS_BRIDGE = True
except ImportError:
    HAS_BRIDGE = False

try:
    from picamera2 import Picamera2
    HAS_PICAMERA = True
except ImportError:
    HAS_PICAMERA = False


# --------- Parametros geometricos / camara ----------------------------------
CUBO_LADO_M   = 0.15
FOCAL_PIXELES = 720.0       # <<< CALIBRAR
FRAME_W       = 640
FRAME_H       = 480
HFOV_RAD      = math.radians(66.0)

# --------- Rangos HSV -------------------------------------------------------
HSV_RANGES = {
    'RED':   [((0,   110,  80), (10,  255, 255)),
              ((170, 110,  80), (180, 255, 255))],
    'GREEN': [((40,   80,  60), (85,  255, 255))],
    'BLUE':  [((95,  110,  60), (130, 255, 255))],
}
MIN_AREA_PX = 1500

# Colores definidos pre-conversion: la imagen se invierte BGR<->RGB en imshow,
# asi que aqui ponemos los rectangulos al reves (R en posicion B y viceversa).
BOX_COLOR = {'RED': (255, 0, 0), 'GREEN': (0, 255, 0), 'BLUE': (0, 0, 255)}

WINDOW_NAME = 'Cubos R/G/B (vision_node)'


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # Parametro para desactivar la ventana si quieres correr "headless"
        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter(
            'show_window').get_parameter_value().bool_value

        self.pub_det = self.create_publisher(String, '/cube_detected', 10)
        self.pub_img = self.create_publisher(Image,  '/vision/image',  10)
        self.bridge  = CvBridge() if HAS_BRIDGE else None

        # --- Camara ---
        self.picam = None
        self.cap   = None
        if HAS_PICAMERA:
            self.picam = Picamera2()
            cfg = self.picam.create_preview_configuration(
                main={'size': (FRAME_W, FRAME_H), 'format': 'RGB888'})
            self.picam.configure(cfg)
            self.picam.start()
            time.sleep(0.5)
            self.get_logger().info("Camara: picamera2 (Camera Module 3)")
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            self.get_logger().info("Camara: cv2.VideoCapture(0)")

        if self.show_window:
            self.get_logger().info(
                "Mostrando ventana OpenCV.  Conecta por 'ssh -X' para verla en tu PC.")
            self.get_logger().info(
                "Pulsa 'q' en la ventana para cerrarla (el nodo sigue corriendo).")
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        self.create_timer(1.0 / 15.0, self._loop)

    # ------------------------------------------------------------------ grab
    def _grab(self):
        if self.picam is not None:
            return cv2.cvtColor(self.picam.capture_array(), cv2.COLOR_RGB2BGR)
        ok, frame = self.cap.read()
        return frame if ok else None

    # --------------------------------------------------------- find one color
    @staticmethod
    def _find_cube(hsv, ranges):
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, np.array(lo), np.array(hi))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < MIN_AREA_PX:
            return None
        return cv2.boundingRect(c)

    # --------------------------------------------------------------- main loop
    def _loop(self):
        frame = self._grab()
        if frame is None:
            return
        # Frame de VideoCapture llega en RGB (no BGR) en esta camara
        hsv  = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        draw = frame.copy()

        best = None
        for color, ranges in HSV_RANGES.items():
            r = self._find_cube(hsv, ranges)
            if r is None:
                continue
            x, y, w, h = r
            # Dibuja TODOS los detectados, no solo el mas grande
            bgr = BOX_COLOR.get(color, (255, 255, 255))
            cv2.rectangle(draw, (x, y), (x + w, y + h), bgr, 2)
            area = w * h
            cx_box = x + w / 2.0
            w_px = max(w, h)
            dist = (FOCAL_PIXELES * CUBO_LADO_M) / max(1.0, w_px)
            cv2.putText(draw, f"{color} {dist:.2f}m", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
            if best is None or area > best[1]:
                best = (color, area, r, dist, cx_box)

        # Publicar /cube_detected (solo el mas grande)
        msg = String()
        if best is None:
            msg.data = "NONE,0,0,0,0.0,0.0"
        else:
            color, _, (x, y, w, h), dist, cx = best
            w_px = max(w, h)
            ang  = -((cx - FRAME_W / 2.0) / FRAME_W) * HFOV_RAD
            msg.data = (f"{color},{int(cx)},{int(y + h / 2)},"
                        f"{int(w_px)},{dist:.3f},{ang:.4f}")
        self.pub_det.publish(msg)

        # Linea central de referencia
        cx_frame = FRAME_W // 2
        cv2.line(draw, (cx_frame, 0), (cx_frame, FRAME_H), (200, 200, 200), 1)

        # Publicar imagen ROS (para rqt_image_view si se usa en otro PC)
        if self.bridge is not None:
            try:
                img_msg = self.bridge.cv2_to_imgmsg(draw, encoding='bgr8')
                img_msg.header.stamp = self.get_clock().now().to_msg()
                self.pub_img.publish(img_msg)
            except Exception:
                pass

        # Mostrar la ventana OpenCV (visible vía SSH X11)
        # Convierte BGR2RGB para que los colores se vean correctos via X11/VcXsrv
        if self.show_window:
            cv2.imshow(WINDOW_NAME, cv2.cvtColor(draw, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

    def destroy_node(self):
        try:
            if self.show_window:
                cv2.destroyAllWindows()
            if self.picam is not None:
                self.picam.stop()
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
