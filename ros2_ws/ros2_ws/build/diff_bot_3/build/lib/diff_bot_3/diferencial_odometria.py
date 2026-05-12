import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import time

class OdometryNode(Node):
    def __init__(self):
        super().__init__('odometry_node')
        self.sub = self.create_subscription(Twist, 'turtlebot_cmdVel', self.update_callback, 10)
        self.pub = self.create_publisher(Twist, 'turtlebot_position', 10)
        
        # Estado del robot
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        self.last_time = time.time()

    def update_callback(self, msg):
        v = msg.linear.x
        w = msg.angular.z
        
        dt = time.time() - self.last_time
        self.last_time = time.time()

        # Ecuaciones Cinemáticas Diferenciales (Criterio 9)
        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += w * dt

        # Publicar posición para la interfaz (Criterio 11)
        pos = Twist()
        pos.linear.x = self.x
        pos.linear.y = self.y
        pos.angular.z = self.theta
        self.pub.publish(pos)

def main():
    rclpy.init()
    rclpy.spin(OdometryNode())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
