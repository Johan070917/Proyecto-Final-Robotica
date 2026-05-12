"""
robot_teleop.py  -  Teleoperacion por teclado (WASD).
Se mantiene como respaldo del joystick bluetooth.
"""

import sys, select, termios, tty
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


BIND = {
    'w': ( 1.0,  0.0),
    's': (-1.0,  0.0),
    'a': ( 0.0,  1.0),
    'd': ( 0.0, -1.0),
}


def get_key(settings, timeout=0.1):
    tty.setraw(sys.stdin.fileno())
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if r else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = rclpy.create_node('robot_teleop')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)

    print("--- TELEOP TECLADO ---")
    v_max = float(input("Velocidad lineal maxima [m/s]: ") or 0.35)
    w_max = float(input("Velocidad angular maxima [rad/s]: ") or 1.5)
    print("WASD para moverse, 'q' para salir.")

    was_moving = False
    try:
        while True:
            k = get_key(settings)
            if k == 'q':
                break
            if k in BIND:
                v_k, w_k = BIND[k]
                msg = Twist()
                msg.linear.x = v_k * v_max
                msg.angular.z = w_k * w_max
                pub.publish(msg)
                was_moving = True
            else:
                if was_moving:
                    pub.publish(Twist())
                    was_moving = False
    finally:
        pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
