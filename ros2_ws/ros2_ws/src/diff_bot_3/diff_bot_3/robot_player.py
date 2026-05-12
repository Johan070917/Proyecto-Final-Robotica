"""
robot_player.py  -  Reproduce una trayectoria grabada usando un controlador
en LAZO CERRADO basado en la odometria.  Este es el cambio clave: ya no
reenvia los comandos v/w originales (lazo abierto, deriva con cualquier
perturbacion), sino que sigue las POSES (x,y,theta) con retroalimentacion.

Controlador: "unicycle trajectory tracking" estandar (Kanayama):
  xe =  cos(th) * (xr - x) + sin(th) * (yr - y)
  ye = -sin(th) * (xr - x) + cos(th) * (yr - y)
  the = atan2(sin(thr - th), cos(thr - th))
  v = vr * cos(the) + Kx * xe
  w = wr + vr * (Ky * ye + Kth * sin(the))

El referencial en el instante t se obtiene por interpolacion lineal
entre muestras del archivo.
"""

import os
import math
import time
import threading

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String


# Ganancias del controlador (ajustables)
K_X   = 1.2
K_Y   = 3.5
K_TH  = 2.0


def yaw_from_quat(qz, qw):
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('robot_player')

        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel_player', 10)
        self.create_subscription(Odometry, '/odom', self._on_odom, 50)
        self.create_subscription(String, '/player_cmd', self._on_player_cmd, 10)

        self.x, self.y, self.theta = 0.0, 0.0, 0.0

        self._playing = False
        self._play_thread = None
        self._traj = []   # lista de (t, x, y, th, v, w)

        self.get_logger().info('robot_player listo (lazo cerrado).')

    def _on_odom(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = yaw_from_quat(
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)

    def _on_player_cmd(self, msg: String):
        data = msg.data.strip()
        if data == 'STOP':
            self._playing = False
            return
        if data.startswith('PLAY:'):
            fname = data.split(':', 1)[1]
            self._start_play(fname)

    # ----------------------- Carga y reproduccion --------------------------
    def _load(self, path):
        traj = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split(',')
                if len(parts) < 4:
                    continue
                t = float(parts[0]); x = float(parts[1]); y = float(parts[2])
                th = float(parts[3])
                v  = float(parts[4]) if len(parts) > 4 else 0.0
                w  = float(parts[5]) if len(parts) > 5 else 0.0
                traj.append((t, x, y, th, v, w))
        return traj

    def _start_play(self, fname):
        if self._playing:
            self.get_logger().warn("Ya hay una reproduccion en curso.")
            return
        path = os.path.join(os.getcwd(), fname)
        if not os.path.exists(path):
            self.get_logger().error(f"No existe el archivo: {path}")
            return
        try:
            self._traj = self._load(path)
        except Exception as e:
            self.get_logger().error(f"Error leyendo {path}: {e}")
            return
        if len(self._traj) < 2:
            self.get_logger().error("Trayectoria demasiado corta.")
            return

        self._playing = True
        self._play_thread = threading.Thread(target=self._run, daemon=True)
        self._play_thread.start()

    # ------------ Interpolacion de la trayectoria de referencia -----------
    def _ref_at(self, t):
        traj = self._traj
        if t <= traj[0][0]:
            return traj[0]
        if t >= traj[-1][0]:
            return traj[-1]
        # busqueda lineal (la trayectoria suele caber en memoria)
        lo, hi = 0, len(traj) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if traj[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = traj[lo], traj[hi]
        dt = b[0] - a[0]
        alpha = 0.0 if dt <= 0 else (t - a[0]) / dt
        xr  = a[1] + alpha * (b[1] - a[1])
        yr  = a[2] + alpha * (b[2] - a[2])
        # interpolar angulo con cuidado (evitar saltos +-pi)
        dth = math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
        thr = a[3] + alpha * dth
        vr  = a[4] + alpha * (b[4] - a[4])
        wr  = a[5] + alpha * (b[5] - a[5])
        return (t, xr, yr, thr, vr, wr)

    # ------------------ Lazo de control de seguimiento --------------------
    def _run(self):
        self.get_logger().info(
            f"Reproduciendo trayectoria ({len(self._traj)} muestras, "
            f"{self._traj[-1][0]:.1f} s)")

        dt_loop = 0.04       # 25 Hz
        t_start = time.monotonic()
        t_final = self._traj[-1][0]

        try:
            while self._playing and rclpy.ok():
                t_play = time.monotonic() - t_start
                if t_play > t_final + 0.3:
                    break
                _, xr, yr, thr, vr, wr = self._ref_at(t_play)

                # errores expresados en el frame del robot
                dx = xr - self.x
                dy = yr - self.y
                c, s = math.cos(self.theta), math.sin(self.theta)
                xe =  c * dx + s * dy
                ye = -s * dx + c * dy
                the = math.atan2(
                    math.sin(thr - self.theta),
                    math.cos(thr - self.theta))

                v = vr * math.cos(the) + K_X * xe
                w = wr + vr * (K_Y * ye + K_TH * math.sin(the))

                # Saturacion fisica
                v = max(-0.5, min(0.5, v))
                w = max(-2.0, min(2.0, w))

                cmd = Twist()
                cmd.linear.x  = v
                cmd.angular.z = w
                self.pub_cmd.publish(cmd)
                time.sleep(dt_loop)
        finally:
            self.pub_cmd.publish(Twist())   # freno
            self._playing = False
            self.get_logger().info("Reproduccion finalizada.")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
