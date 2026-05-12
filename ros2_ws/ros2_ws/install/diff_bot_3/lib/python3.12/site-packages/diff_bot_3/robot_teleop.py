import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys, select, termios, tty

moveBindings = {'w': (1, 0), 's': (-1, 0), 'a': (0, 1), 'd': (0, -1)}

def get_key(settings):
    tty.setraw(sys.stdin.fileno())
    # Espera 0.1s. Si no hay tecla, devuelve string vacío
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def main():
    settings = termios.tcgetattr(sys.stdin)
    rclpy.init()
    node = rclpy.create_node('teleop_node')
    pub = node.create_publisher(Twist, 'cmd_vel', 10)

    print("--- CONTROL LISTO ---")
    v_limit = float(input("Velocidad Lineal: "))
    w_limit = float(input("Velocidad Angular: "))
    print("Mueve con WASD (Mantén presionado). 'q' para salir.")

    try:
        was_moving = False
        while True:
            key = get_key(settings)
            msg = Twist()
            
            if key in moveBindings.keys():
                msg.linear.x = moveBindings[key][0] * v_limit
                msg.angular.z = moveBindings[key][1] * w_limit
                pub.publish(msg)
                was_moving = True
            elif key == 'q':
                break
            else:
                # Si no hay tecla y antes se movía, frena una vez
                if was_moving:
                    pub.publish(Twist()) # Envía todo en 0.0
                    was_moving = False
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pub.publish(Twist())
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()