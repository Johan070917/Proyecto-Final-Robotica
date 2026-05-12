import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
import time
import os

class PlayerDiferencial(Node):
    def __init__(self):
        super().__init__('player_diferencial')
        
        # Publicador conectado a los motores reales
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.srv = self.create_service(Trigger, 'play_recording', self.play_callback)
        self.get_logger().info('Nodo Player listo y esperando servicio...')

    def play_callback(self, request, response):
        try:
            ptr_path = os.path.join(os.getcwd(), "last_file.ptr")
            with open(ptr_path, "r") as f:
                filename = f.read().strip()
            
            file_path = os.path.join(os.getcwd(), f"{filename}.txt")
            
            if not os.path.exists(file_path):
                response.success = False
                response.message = f"Error: El archivo {filename}.txt no existe."
                return response

            self.get_logger().info(f'Reproduciendo trayectoria de: {filename}.txt')
            
            with open(file_path, "r") as f:
                lines = f.readlines()
                
            start_play_time = time.time()
            for line in lines:
                t_rec, v, delta = map(float, line.split(','))
                
                while (time.time() - start_play_time) < t_rec:
                    time.sleep(0.001)
                
                msg = Twist()
                msg.linear.x = v
                msg.angular.z = delta
                self.publisher.publish(msg)

            # Frenado de seguridad
            stop_msg = Twist()
            self.publisher.publish(stop_msg)
            
            response.success = True
            response.message = "Reproducción completada con éxito."
            
        except Exception as e:
            response.success = False
            response.message = f"Error durante la reproducción: {str(e)}"
            
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PlayerDiferencial()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
