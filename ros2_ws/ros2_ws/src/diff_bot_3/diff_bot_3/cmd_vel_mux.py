"""
cmd_vel_mux.py  -  Multiplexor de /cmd_vel.

Prioridades (mayor gana):
  1. /cmd_vel_teleop  (joystick / teclado)  -  manual siempre prevalece
                                              SI esta moviendo realmente
  2. /cmd_vel_auto    (forklift_manager: secuencia D-pad)
  3. /cmd_vel_player  (reproductor de trayectoria)

Detalle clave: teleop_twist_joy publica continuamente (incluso ceros
cuando el stick esta en reposo). Si simplemente diéramos prioridad a
"teleop fresco", el cmd_vel_auto del forklift_manager NUNCA ganaria
mientras el joystick este conectado, porque teleop siempre estaria
"fresco". Por eso aqui distinguimos entre "fresco" y "ACTIVO":
  fresco  = recibido en los ultimos 0.3 s
  activo  = fresco Y con linear/angular distintos de cero

Politica:
  - Si teleop esta ACTIVO  -> teleop manda (anula auto y player)
  - Si auto   esta fresco  -> auto manda    (forklift en plena secuencia)
  - Si player esta fresco  -> player manda
  - Si ninguno -> Twist() (parar)
"""

import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# Umbrales para considerar "activo" (no cero) un comando teleop.
# Por debajo de esto se considera "stick en reposo, sin intencion".
EPS_LIN = 0.02
EPS_ANG = 0.05


class CmdVelMux(Node):
    def __init__(self):
        super().__init__('cmd_vel_mux')
        self.last = {
            'teleop': (0.0, Twist()),
            'auto':   (0.0, Twist()),
            'player': (0.0, Twist()),
        }
        self.create_subscription(Twist, '/cmd_vel_teleop', self._cb('teleop'), 10)
        self.create_subscription(Twist, '/cmd_vel_auto',   self._cb('auto'),   10)
        self.create_subscription(Twist, '/cmd_vel_player', self._cb('player'), 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_timer(0.05, self._publish)
        self.get_logger().info("cmd_vel_mux activo.")

    def _cb(self, key):
        def cb(msg: Twist):
            self.last[key] = (time.monotonic(), msg)
        return cb

    def _fresh(self, key, thresh=0.3):
        t, _ = self.last[key]
        return (time.monotonic() - t) < thresh

    def _is_nonzero(self, msg: Twist) -> bool:
        return (abs(msg.linear.x)  > EPS_LIN or
                abs(msg.linear.y)  > EPS_LIN or
                abs(msg.angular.z) > EPS_ANG)

    def _teleop_active(self) -> bool:
        if not self._fresh('teleop'):
            return False
        return self._is_nonzero(self.last['teleop'][1])

    def _publish(self):
        # 1) Joystick activo (intencion manual) -> manda joystick
        if self._teleop_active():
            self.pub.publish(self.last['teleop'][1])
            return
        # 2) Forklift haciendo secuencia (avanzar/retroceder 21 cm) -> auto
        if self._fresh('auto'):
            self.pub.publish(self.last['auto'][1])
            return
        # 3) Reproductor de trayectoria -> player
        if self._fresh('player'):
            self.pub.publish(self.last['player'][1])
            return
        # 4) Nada activo -> parar
        self.pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
