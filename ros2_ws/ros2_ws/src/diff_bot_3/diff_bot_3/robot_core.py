"""
robot_core.py  -  Nodo "maestro" de bajo nivel del robot diferencial.

Arquitectura nueva (master-slave con dos ESP32):
  * RPi5 (este nodo) calcula odometria + PID de velocidad por rueda.
  * ESP32_MOTORS aplica el PWM a los motores y reporta encoders.

Topicos:
  Sub:  /motors_status (std_msgs/String)   <- "ENC <l> <r> <dt_us>"
  Sub:  /cmd_vel       (geometry_msgs/Twist)
  Pub:  /motors_cmd    (std_msgs/String)   -> "PWM <left> <right>"
  Pub:  /odom          (nav_msgs/Odometry)
  Pub:  /wheel_speeds  (std_msgs/Float32MultiArray) [v_l, v_r, v_ref_l, v_ref_r]
"""

import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String


# --------------------------- PARAMETROS FISICOS -----------------------------
R_RUEDA          = 0.060      # Radio de rueda [m]
L_BASE           = 0.209      # Distancia entre ruedas [m]
TICKS_POR_VUELTA = 374.0      # Pulsos de encoder por vuelta de rueda

# Si un encoder cuenta al reves respecto al sentido fisico, pon -1 aqui
SIGN_LEFT        = +1
SIGN_RIGHT       = +1

# Velocidad maxima real (m/s) a PWM=1.0.  Calibrar.
V_MAX_RUEDA      = 0.30

# PID por rueda
KP_V = 1.80
KI_V = 4.20
KD_V = 0.00
I_MAX = 0.80

# Periodo del lazo de control en RPi
DT_CONTROL = 0.02            # 50 Hz

# Rampa de aceleracion: limite cuanto puede CAMBIAR la velocidad de
# referencia por segundo (slew rate). Sin esto, soltar el stick a fondo
# manda al PID a saltar de 0 a v_max instantaneamente y el robot
# tiende a caerse hacia atras por inercia (centro de masa alto).
# Valores tipicos: 0.3-0.6 m/s^2 para arranques suaves.
ACC_LIN_MAX_MPS2 = 0.40   # m/s^2
ACC_ANG_MAX_RPS2 = 2.50   # rad/s^2


# ----------------------------------- UTIL -----------------------------------
def quat_from_yaw(yaw):
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class PID:
    def __init__(self, kp, ki, kd, i_max=I_MAX):
        self.kp, self.ki, self.kd, self.i_max = kp, ki, kd, i_max
        self.i = 0.0
        self.prev_e = 0.0

    def step(self, err, dt):
        self.i = max(-self.i_max, min(self.i_max, self.i + err * dt))
        d = (err - self.prev_e) / dt if dt > 0 else 0.0
        self.prev_e = err
        return self.kp * err + self.ki * self.i + self.kd * d

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0


# ----------------------------------- NODO -----------------------------------
class RobotCore(Node):
    def __init__(self):
        super().__init__('robot_core')

        # Pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Encoders
        self.last_ticks_l = None
        self.last_ticks_r = None
        self.v_real_l = 0.0
        self.v_real_r = 0.0
        self.last_enc_time = None

        # Referencia (la pedida por el usuario / forklift)
        self.v_ref = 0.0
        self.w_ref = 0.0
        # Referencia "filtrada" por la rampa de aceleracion (la que
        # realmente alimenta el PID).
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self._last_cmd_time = self.get_clock().now()

        # PID
        self.pid_l = PID(KP_V, KI_V, KD_V)
        self.pid_r = PID(KP_V, KI_V, KD_V)

        # ROS I/O
        self.create_subscription(Twist,  '/cmd_vel',        self._on_cmd_vel,  10)
        self.create_subscription(String, '/motors_status',  self._on_motors,   50)
        self.pub_motors_cmd = self.create_publisher(String, '/motors_cmd', 10)
        self.pub_odom       = self.create_publisher(Odometry, '/odom', 50)
        self.pub_wheels     = self.create_publisher(Float32MultiArray, '/wheel_speeds', 10)

        self.create_timer(DT_CONTROL, self._control_loop)
        self.get_logger().info("robot_core listo (PID en RPi a 50 Hz, slave=ESP32_MOTORS).")

    # ------------------------- Callbacks ---------------------------------
    def _on_cmd_vel(self, msg: Twist):
        # Motores cableados al reves respecto a la convencion: invertir v_ref
        # para que stick adelante = robot adelante. Angular se deja igual.
        self.v_ref = -float(msg.linear.x)
        self.w_ref = float(msg.angular.z)
        self._last_cmd_time = self.get_clock().now()

    def _on_motors(self, msg: String):
        # Espera "ENC <left> <right> <dt_us>"
        parts = msg.data.strip().split()
        if len(parts) != 4 or parts[0] != 'ENC':
            return
        try:
            ticks_l = int(parts[1]) * SIGN_LEFT
            ticks_r = int(parts[2]) * SIGN_RIGHT
            dt_us   = int(parts[3])
        except ValueError:
            return

        if self.last_ticks_l is None:
            self.last_ticks_l = ticks_l
            self.last_ticks_r = ticks_r
            return

        d_ticks_l = ticks_l - self.last_ticks_l
        d_ticks_r = ticks_r - self.last_ticks_r
        self.last_ticks_l = ticks_l
        self.last_ticks_r = ticks_r

        dt = max(dt_us, 1) / 1.0e6

        # Distancias por rueda
        d_l = 2.0 * math.pi * R_RUEDA * (d_ticks_l / TICKS_POR_VUELTA)
        d_r = 2.0 * math.pi * R_RUEDA * (d_ticks_r / TICKS_POR_VUELTA)

        # Velocidades reales
        self.v_real_l = d_l / dt
        self.v_real_r = d_r / dt

        # Odometria
        d_c  = (d_l + d_r) / 2.0
        d_th = (d_r - d_l) / L_BASE
        self.x += d_c * math.cos(self.theta + d_th / 2.0)
        self.y += d_c * math.sin(self.theta + d_th / 2.0)
        self.theta = math.atan2(math.sin(self.theta + d_th),
                                math.cos(self.theta + d_th))

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = quat_from_yaw(self.theta)
        odom.twist.twist.linear.x  = d_c / dt
        odom.twist.twist.angular.z = d_th / dt
        self.pub_odom.publish(odom)

    # ------------------------- Lazo de control -------------------------
    def _control_loop(self):
        # Watchdog del comando
        age = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
        if age > 0.5:
            self.v_ref = 0.0
            self.w_ref = 0.0
            self.pid_l.reset()
            self.pid_r.reset()

        # Si la referencia es esencialmente cero (joystick en reposo),
        # frena duro y resetea integral. Sin esto, el drift del stick
        # acumula integral hasta superar la zona muerta y dispara el motor.
        if abs(self.v_ref) < 0.02 and abs(self.w_ref) < 0.05:
            self.pid_l.reset()
            self.pid_r.reset()
            # Tambien resetear las referencias filtradas para que al
            # arrancar de nuevo, la rampa empiece desde 0.
            self.v_cmd = 0.0
            self.w_cmd = 0.0
            self.pub_motors_cmd.publish(String(data="PWM 0 0"))
            wm = Float32MultiArray()
            wm.data = [self.v_real_l, self.v_real_r, 0.0, 0.0]
            self.pub_wheels.publish(wm)
            return

        # Rampa de aceleracion: aproxima v_cmd a v_ref con un cambio
        # maximo de ACC_LIN_MAX_MPS2 * DT por ciclo. Esto evita que la
        # velocidad pedida salte de 0 a max al apretar el stick.
        dv_max = ACC_LIN_MAX_MPS2 * DT_CONTROL
        dw_max = ACC_ANG_MAX_RPS2 * DT_CONTROL
        self.v_cmd += max(-dv_max, min(dv_max, self.v_ref - self.v_cmd))
        self.w_cmd += max(-dw_max, min(dw_max, self.w_ref - self.w_cmd))

        # Cinematica inversa (usando la referencia filtrada por la rampa)
        v_ref_l = self.v_cmd + (self.w_cmd * L_BASE / 2.0)
        v_ref_r = self.v_cmd - (self.w_cmd * L_BASE / 2.0)

        # Feed-forward + PID
        ff_l = v_ref_l / V_MAX_RUEDA
        ff_r = v_ref_r / V_MAX_RUEDA
        u_l = ff_l + self.pid_l.step(v_ref_l - self.v_real_l, DT_CONTROL)
        u_r = ff_r + self.pid_r.step(v_ref_r - self.v_real_r, DT_CONTROL)

        # Saturacion + zona muerta
        u_l = max(-1.0, min(1.0, u_l))
        u_r = max(-1.0, min(1.0, u_r))
        if abs(u_l) < 0.08: u_l = 0.0
        if abs(u_r) < 0.08: u_r = 0.0

        # Convertir a PWM int [-255, 255]
        pwm_l = int(round(u_l * 255.0))
        pwm_r = int(round(u_r * 255.0))

        # Mandar a la ESP32
        self.pub_motors_cmd.publish(
            String(data=f"PWM {pwm_l} {pwm_r}"))

        # Telemetria
        wm = Float32MultiArray()
        wm.data = [self.v_real_l, self.v_real_r, v_ref_l, v_ref_r]
        self.pub_wheels.publish(wm)

    def destroy_node(self):
        try:
            self.pub_motors_cmd.publish(String(data="STOP"))
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RobotCore()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
