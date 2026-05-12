"""
forklift_manager.py  -  Logica autonoma del montacargas.

Secuencia disparada por el D-pad del joystick:

  D-PAD ARRIBA (axes[7] > 0.5):
      1. FORKS_OPEN  (servo gira: pala BAJA = horizontal / paralela al suelo)
      2. Avanza 21 cm en linea recta usando /odom
      3. LIFT_UP <ticks>  (sube el montacargas con el motor 25GA370)
      4. Queda en estado HOLDING (el operador conduce con el joystick)

  D-PAD ABAJO (axes[7] < -0.5):
      1. LIFT_DOWN <ticks>  (baja el montacargas)
      2. Retrocede 21 cm en linea recta usando /odom
      3. FORKS_CLOSE  (servo gira: pala SUBE = vertical / recogida)
      4. Vuelve a IDLE

Comandos al lift por /lift_cmd (los maneja la ESP32_LIFT).
Acks de la ESP32 llegan por /lift_status.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy


# ============== PARAMETROS (ajustar con calibracion real) =================
DIST_FORWARD_M    = 0.21    # 21 cm hacia adelante al activar D-PAD UP
DIST_BACKWARD_M   = 0.21    # 21 cm hacia atras al activar D-PAD DOWN
V_DRIVE           = 0.08    # m/s durante avance / retroceso autonomo
W_HEADING_GAIN    = 1.5     # P sobre el yaw para mantener rumbo

# Ticks del encoder del 25GA370 para subir/bajar el montacargas.
# CALIBRAR enviando LIFT_UP 100, midiendo cuantos mm sube y escalando.
LIFT_TICKS_FULL   = 4000

# Timeouts
T_TIMEOUT_OP      = 15.0    # operacion individual ESP32 (lift / servo)
T_TIMEOUT_DRIVE   = 8.0     # avance / retroceso de 21 cm


# =======================================================================
def yaw_from_quat(qz, qw):
    return math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz)


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


class ForkliftManager(Node):
    def __init__(self):
        super().__init__('forklift_manager')

        # FSM
        self.state   = 'IDLE'
        self.t_state = time.monotonic()

        # Odometria
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.odom_ready = False

        # Referencia al iniciar avance / retroceso
        self.x0 = 0.0
        self.y0 = 0.0
        self.theta0 = 0.0
        self.target_dist = 0.0
        self.drive_sign  = +1

        # Acks de la ESP32_LIFT
        self.lift_done = False
        self.lift_err  = False

        # Flanco del D-pad
        self.last_dpad_up   = 0
        self.last_dpad_down = 0

        # Cubo detectado (solo log)
        self.last_color = 'NONE'

        # Topicos
        self.create_subscription(Odometry, '/odom',          self._on_odom,  50)
        self.create_subscription(String,   '/cube_detected', self._on_cube,  10)
        self.create_subscription(String,   '/lift_status',   self._on_lift,  10)
        self.create_subscription(Joy,      '/joy',           self._on_joy,   10)

        self.pub_lift_cmd = self.create_publisher(String, '/lift_cmd',     10)
        self.pub_vel      = self.create_publisher(Twist,  '/cmd_vel_auto', 10)

        self.create_timer(0.05, self._tick)        # 20 Hz
        self.get_logger().info("forklift_manager listo.  Estado = IDLE")
        self.get_logger().info(
            "D-PAD ARRIBA = recoger cubo (FORKS_OPEN -> avanzar 21 cm -> subir lift)")
        self.get_logger().info(
            "D-PAD ABAJO  = soltar cubo (bajar lift -> retroceder 21 cm -> FORKS_CLOSE)")

    # ------------------------- CALLBACKS ------------------------------
    def _on_odom(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.theta = yaw_from_quat(
            msg.pose.pose.orientation.z, msg.pose.pose.orientation.w)
        self.odom_ready = True

    def _on_cube(self, msg: String):
        parts = msg.data.split(',')
        if parts and parts[0] in ('RED', 'GREEN', 'BLUE'):
            self.last_color = parts[0]

    def _on_lift(self, msg: String):
        data = msg.data.strip()
        if data == 'DONE':
            self.lift_done = True
        elif data.startswith('ERR'):
            self.lift_err = True
            self.get_logger().warn(f"ESP32_LIFT error: {data}")

    def _on_joy(self, msg: Joy):
        # D-pad vertical normalmente esta en axes[7]: +1 arriba, -1 abajo.
        if len(msg.axes) <= 7:
            return
        dpad_v    = msg.axes[7]
        dpad_up   = 1 if dpad_v >  0.5 else 0
        dpad_down = 1 if dpad_v < -0.5 else 0

        if dpad_up and not self.last_dpad_up and self.state == 'IDLE':
            self.get_logger().info(
                f"D-PAD ARRIBA: iniciando recogida (cubo: {self.last_color})")
            self._go('OPEN_FORKS')
        elif dpad_down and not self.last_dpad_down and self.state == 'HOLDING':
            self.get_logger().info("D-PAD ABAJO: iniciando deposito")
            self._go('LIFT_DOWN_OP')

        self.last_dpad_up   = dpad_up
        self.last_dpad_down = dpad_down

    # ------------------------- HELPERS --------------------------------
    def _go(self, new_state: str):
        self.get_logger().info(f"Estado: {self.state} -> {new_state}")
        self.state = new_state
        self.t_state = time.monotonic()
        self.lift_done = False
        self.lift_err  = False

    def _send_lift(self, cmd: str):
        self.pub_lift_cmd.publish(String(data=cmd))

    def _begin_drive(self, sign: int, dist: float, next_state: str):
        if not self.odom_ready:
            self.get_logger().warn("Odometria no lista; abortando.")
            self._go('IDLE')
            return
        self.x0 = self.x
        self.y0 = self.y
        self.theta0 = self.theta
        self.target_dist = abs(dist)
        self.drive_sign  = sign
        self._go(next_state)

    def _drive_step(self) -> bool:
        # Avanza/retrocede en linea recta hasta cubrir target_dist desde (x0,y0).
        dx = self.x - self.x0
        dy = self.y - self.y0
        traveled = math.hypot(dx, dy)

        if traveled >= self.target_dist:
            self.pub_vel.publish(Twist())
            return True

        e_th = angle_diff(self.theta0, self.theta)
        cmd = Twist()
        cmd.linear.x  = self.drive_sign * V_DRIVE
        cmd.angular.z = self.drive_sign * W_HEADING_GAIN * e_th
        self.pub_vel.publish(cmd)
        return False

    # --------------------------- FSM ----------------------------------
    def _tick(self):
        age = time.monotonic() - self.t_state

        if self.state == 'IDLE':
            return

        # ----- D-PAD UP : OPEN_FORKS -> DRIVE_FORWARD -> LIFT_UP -> HOLDING -----

        elif self.state == 'OPEN_FORKS':
            if age < 0.05:
                self._send_lift('FORKS_OPEN')
                return
            if self.lift_done:
                self._begin_drive(+1, DIST_FORWARD_M, next_state='DRIVE_FORWARD')
            elif age > T_TIMEOUT_OP or self.lift_err:
                self.get_logger().warn("Timeout abriendo palas. Aborta.")
                self._go('IDLE')

        elif self.state == 'DRIVE_FORWARD':
            if age > T_TIMEOUT_DRIVE:
                self.pub_vel.publish(Twist())
                self.get_logger().warn("Timeout avanzando. Aborta.")
                self._go('IDLE')
                return
            if self._drive_step():
                self._go('LIFT_UP_OP')

        elif self.state == 'LIFT_UP_OP':
            if age < 0.05:
                self._send_lift(f'LIFT_UP {LIFT_TICKS_FULL}')
                return
            if self.lift_done:
                self.get_logger().info(
                    f"Montacargas arriba.  Cubo cargado: {self.last_color}")
                self._go('HOLDING')
            elif age > T_TIMEOUT_OP * 2 or self.lift_err:
                self.get_logger().warn("Falla subiendo el lift. Pasando a HOLDING.")
                self._go('HOLDING')

        elif self.state == 'HOLDING':
            return

        # ----- D-PAD DOWN : LIFT_DOWN_OP -> DRIVE_BACK -> CLOSE_FORKS -> IDLE -

        elif self.state == 'LIFT_DOWN_OP':
            if age < 0.05:
                self._send_lift(f'LIFT_DOWN {LIFT_TICKS_FULL}')
                return
            if self.lift_done:
                self._begin_drive(-1, DIST_BACKWARD_M, next_state='DRIVE_BACK')
            elif age > T_TIMEOUT_OP * 2 or self.lift_err:
                self.get_logger().warn("Falla bajando el lift. Continua igual.")
                self._begin_drive(-1, DIST_BACKWARD_M, next_state='DRIVE_BACK')

        elif self.state == 'DRIVE_BACK':
            if age > T_TIMEOUT_DRIVE:
                self.pub_vel.publish(Twist())
                self.get_logger().warn("Timeout retrocediendo.")
                self._go('CLOSE_FORKS')
                return
            if self._drive_step():
                self._go('CLOSE_FORKS')

        elif self.state == 'CLOSE_FORKS':
            if age < 0.05:
                self._send_lift('FORKS_CLOSE')
                return
            if self.lift_done or age > T_TIMEOUT_OP or self.lift_err:
                self.get_logger().info("Secuencia de deposito completada.")
                self._go('IDLE')


def main(args=None):
    rclpy.init(args=args)
    node = ForkliftManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
