"""
robot_interface.py  -  Interfaz grafica (Tk + matplotlib).

Cambios respecto a la version anterior:
  * Graba POSES (t, x, y, theta, v, w) muestreadas a la misma frecuencia
    que /odom.  Asi el reproductor puede cerrar lazo sobre la TRAYECTORIA
    real (no sobre comandos v/w que derivan con cualquier cambio de carga).
  * El dibujo de la trayectoria se refresca a 20 Hz con "blit" de
    matplotlib, sin perder puntos.
  * Botones para INICIAR / DETENER grabacion y CARGAR / REPRODUCIR
    trayectoria (sin input() bloqueante).
"""

import os
import math
import time
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox


class RobotInterface(Node):
    def __init__(self):
        super().__init__('robot_interface')

        # --- Estado de trayectoria en memoria -------------------------------
        self.x_real, self.y_real = deque(maxlen=100_000), deque(maxlen=100_000)
        self.x_ref,  self.y_ref  = [], []          # trayectoria cargada
        self._yaw_last = 0.0
        self._v_last   = 0.0
        self._w_last   = 0.0

        # --- Grabacion ------------------------------------------------------
        self.recording = False
        self.rec_buffer = []       # lista de tuplas (t, x, y, th, v, w)
        self.t0_rec = 0.0

        # --- Publicadores ---------------------------------------------------
        self.pub_play = self.create_publisher(String, '/player_cmd', 10)

        # --- Suscripciones --------------------------------------------------
        self.create_subscription(Odometry, '/odom', self._on_odom, 50)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)

        # --- GUI ------------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        plt.subplots_adjust(bottom=0.28)
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-2.5, 2.5)
        self.ax.set_ylim(-2.5, 2.5)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title('Trayectoria del robot (odometria)')
        self.line_real, = self.ax.plot([], [], 'b-', lw=2.0, label='Real')
        self.line_ref,  = self.ax.plot([], [], 'r--', lw=1.4, label='Trayectoria cargada')
        self.dot_robot, = self.ax.plot([], [], 'go', ms=8)
        self.ax.legend(loc='upper right')

        # Botones
        ax_rec   = plt.axes([0.06, 0.14, 0.18, 0.07])
        ax_stop  = plt.axes([0.27, 0.14, 0.18, 0.07])
        ax_load  = plt.axes([0.06, 0.04, 0.18, 0.07])
        ax_play  = plt.axes([0.27, 0.04, 0.18, 0.07])
        ax_clear = plt.axes([0.76, 0.04, 0.18, 0.07])
        ax_tb    = plt.axes([0.55, 0.04, 0.18, 0.07])

        self.btn_rec   = Button(ax_rec,  'Grabar')
        self.btn_stop  = Button(ax_stop, 'Detener/Guardar')
        self.btn_load  = Button(ax_load, 'Cargar')
        self.btn_play  = Button(ax_play, 'Reproducir')
        self.btn_clear = Button(ax_clear, 'Limpiar')
        self.tb_fname  = TextBox(ax_tb, 'Archivo', initial='trayectoria')

        self.btn_rec.on_clicked(self._on_rec)
        self.btn_stop.on_clicked(self._on_stop)
        self.btn_load.on_clicked(self._on_load)
        self.btn_play.on_clicked(self._on_play)
        self.btn_clear.on_clicked(self._on_clear)

        self.get_logger().info("robot_interface listo.")

    # =========================== CALLBACKS ROS ============================
    def _on_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # yaw desde cuaternion (solo z,w ya que el robot es plano)
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        yaw = math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)
        self._yaw_last = yaw
        self.x_real.append(x)
        self.y_real.append(y)

        if self.recording:
            t = time.monotonic() - self.t0_rec
            self.rec_buffer.append(
                (t, x, y, yaw, self._v_last, self._w_last))

    def _on_cmd(self, msg: Twist):
        # Se guarda la ultima referencia v,w para archivar junto con la pose
        self._v_last = float(msg.linear.x)
        self._w_last = float(msg.angular.z)

    # =========================== BOTONES GUI ==============================
    def _on_rec(self, _):
        self.rec_buffer.clear()
        self.t0_rec = time.monotonic()
        self.recording = True
        self.get_logger().info("Grabacion INICIADA.")

    def _on_stop(self, _):
        if not self.recording:
            return
        self.recording = False
        fname = self._filename()
        path = os.path.join(os.getcwd(), fname)
        try:
            with open(path, 'w') as f:
                f.write("# t,x,y,theta,v_ref,w_ref\n")
                for t, x, y, th, v, w in self.rec_buffer:
                    f.write(f"{t:.4f},{x:.5f},{y:.5f},{th:.5f},{v:.4f},{w:.4f}\n")
            self.get_logger().info(f"Guardado: {path}  ({len(self.rec_buffer)} muestras)")
        except Exception as e:
            self.get_logger().error(f"No se pudo guardar: {e}")

    def _on_load(self, _):
        fname = self._filename()
        path = os.path.join(os.getcwd(), fname)
        try:
            xs, ys = [], []
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    xs.append(float(parts[1]))
                    ys.append(float(parts[2]))
            self.x_ref, self.y_ref = xs, ys
            self.get_logger().info(f"Trayectoria cargada: {path} ({len(xs)} puntos)")
        except Exception as e:
            self.get_logger().error(f"No se pudo cargar: {e}")

    def _on_play(self, _):
        msg = String()
        msg.data = f"PLAY:{self._filename()}"
        self.pub_play.publish(msg)
        self.get_logger().info(f"Solicitada reproduccion de {self._filename()}")

    def _on_clear(self, _):
        self.x_real.clear()
        self.y_real.clear()
        self.x_ref, self.y_ref = [], []

    # =========================== HELPERS ==================================
    def _filename(self) -> str:
        name = (self.tb_fname.text or 'trayectoria').strip()
        if not name.endswith('.txt'):
            name += '.txt'
        return name

    # =========================== DIBUJO ===================================
    def refresh_plot(self, _frame):
        self.line_real.set_data(list(self.x_real), list(self.y_real))
        if self.x_ref:
            self.line_ref.set_data(self.x_ref, self.y_ref)
        if self.x_real:
            self.dot_robot.set_data([self.x_real[-1]], [self.y_real[-1]])
            # auto-ajuste suave
            x_arr = list(self.x_real) + list(self.x_ref)
            y_arr = list(self.y_real) + list(self.y_ref)
            xmin, xmax = min(x_arr), max(x_arr)
            ymin, ymax = min(y_arr), max(y_arr)
            pad = 0.5
            self.ax.set_xlim(xmin - pad, xmax + pad)
            self.ax.set_ylim(ymin - pad, ymax + pad)
        return self.line_real, self.line_ref, self.dot_robot


def main(args=None):
    rclpy.init(args=args)
    node = RobotInterface()
    ani = FuncAnimation(node.fig, node.refresh_plot, interval=50, blit=False, cache_frame_data=False)
    t = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    t.start()
    try:
        plt.show()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
