import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry # Importación clave para el robot real
from std_srvs.srv import Trigger
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import threading
import time
import os

class InterfaceDiferencial(Node):
    def __init__(self):
        super().__init__('interface_diferencial')
        
        # Suscripción a la odometría real
        self.sub_pos = self.create_subscription(Odometry, '/odom', self.pos_callback, 10)
        # Suscripción a los comandos para grabación
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_callback, 10)
        
        self.x_data, self.y_data = [], []
        self.log_file = None
        self.start_time = 0

        self.player_client = self.create_client(Trigger, 'play_recording')

        # Configuración de la Gráfica
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        self.line, = self.ax.plot([], [], 'b-', label='Trayectoria Robot Real')
        
        self.ax.set_xlim(-2.0, 2.0)
        self.ax.set_ylim(-2.0, 2.0)
        self.ax.set_title("Visualización en Tiempo Real - Odometría")
        self.ax.legend()

        self.ax_btn = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.btn = Button(self.ax_btn, 'Reproducir')
        self.btn.on_clicked(self.call_player_service)

        self.check_recording()

    def check_recording(self):
        save = input("¿Desea guardar el recorrido del robot? (s/n): ").lower() == 's'
        if save:
            fname = input("Ingrese el nombre del archivo (sin .txt): ")
            path = os.path.join(os.getcwd(), f"{fname}.txt")
            self.log_file = open(path, "w")
            self.start_time = time.time()
            print(f"--- GRABANDO COMANDOS EN {path} ---")

    def pos_callback(self, msg):
        # Extracción correcta de (X,Y) desde nav_msgs/Odometry
        self.x_data.append(msg.pose.pose.position.x)
        self.y_data.append(msg.pose.pose.position.y)

    def cmd_callback(self, msg):
        if self.log_file:
            dt = time.time() - self.start_time
            self.log_file.write(f"{dt:.4f},{msg.linear.x:.4f},{msg.angular.z:.4f}\n")

    def call_player_service(self, event):
        # OJO: Se debe ingresar el texto en la terminal donde lanzaste el nodo
        fname = input("\nNombre del archivo a reproducir: ")
        with open(os.path.join(os.getcwd(), "last_file.ptr"), "w") as f:
            f.write(fname)
        
        if self.player_client.wait_for_service(timeout_sec=1.0):
            req = Trigger.Request()
            self.player_client.call_async(req)
            print(f"Llamando al servicio para reproducir: {fname}...")

    def update_plot(self, frame):
        if self.x_data:
            self.line.set_data(self.x_data, self.y_data)
        return self.line,

def main(args=None):
    rclpy.init(args=args)
    node = InterfaceDiferencial()
    ani = FuncAnimation(node.fig, node.update_plot, interval=100)
    
    thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    plt.show()
    
    if node.log_file:
        node.log_file.close()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
