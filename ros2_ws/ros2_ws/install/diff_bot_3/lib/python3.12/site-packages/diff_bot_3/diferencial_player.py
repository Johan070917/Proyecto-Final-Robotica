import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_srvs.srv import Trigger
import time
import os

class PlayerAckermann(Node):
    def __init__(self):
        super().__init__('player_ackermann')
        
        # Publicador de comandos hacia el robot/odometría
        self.publisher = self.create_publisher(Twist, 'turtlebot_cmdVel', 10)
        
        # Servidor de servicio para iniciar la reproducción (Criterio 13)
        self.srv = self.create_service(Trigger, 'play_recording', self.play_callback)
        
        self.get_logger().info('Nodo Player Ackermann listo y esperando servicio...')

    def play_callback(self, request, response):
        # 1. Leer el nombre del archivo desde el puntero generado por la interfaz
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
            
            # 2. Leer y ejecutar la secuencia de comandos
            with open(file_path, "r") as f:
                lines = f.readlines()
                
            start_play_time = time.time()
            for line in lines:
                # El formato es: tiempo,velocidad,angulo
                t_rec, v, delta = map(float, line.split(','))
                
                # Esperar hasta el momento exacto del comando (reproducción fiel)
                while (time.time() - start_play_time) < t_rec:
                    time.sleep(0.001)
                
                # Publicar el comando
                msg = Twist()
                msg.linear.x = v
                msg.angular.z = delta
                self.publisher.publish(msg)

            # 3. Frenar el robot al terminar
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
    node = PlayerAckermann()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
