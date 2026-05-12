import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from gpiozero import Motor, DigitalInputDevice
import math
import time

class RobotCore(Node):
    def __init__(self):
        super().__init__('robot_core_node')
        
        # --- CONFIGURACIÓN DE MOTORES (gpiozero hace el PWM automático) ---
        # Motor(forward, backward, enable)
        self.motor_izq = Motor(forward=5, backward=6, enable=12)
        self.motor_der = Motor(forward=23, backward=24, enable=13)
        
        # --- CONFIGURACIÓN DE ENCODERS ---
        self.enc_izq_a = DigitalInputDevice(17)
        self.enc_izq_b = DigitalInputDevice(27)
        self.enc_der_a = DigitalInputDevice(22)
        self.enc_der_b = DigitalInputDevice(25)
        
        self.ticks_izq = 0
        self.ticks_der = 0
        
        # Interrupciones (cuando el pin A pasa a estado alto, cuenta un tick)
        self.enc_izq_a.when_activated = self.contar_tick_izq
        self.enc_der_a.when_activated = self.contar_tick_der

        # --- PARÁMETROS FÍSICOS DEL ROBOT ---
        self.R = 0.06 # Radio de la rueda en metros
        self.L = 0.209  # Distancia entre las dos ruedas en metros
        self.TICKS_POR_VUELTA = 374.0 # Ajusta según la reducción de tus motores
        
        # Variables de Odometría
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.last_time = self.get_clock().now()
        
        # --- ROS 2 SUBSCRIBERS Y PUBLISHERS ---
        self.subscription = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.odom_pub = self.create_publisher(Odometry, '/odom', 10)
        
        # Timer para publicar la odometría a 20 Hz
        self.timer = self.create_timer(0.05, self.publicar_odometria)
        
        self.get_logger().info("Robot Core iniciado en la RPi 5. Motores y Encoders listos.")

    def contar_tick_izq(self):
        if self.enc_izq_b.is_active:
            self.ticks_izq -= 1
        else:
            self.ticks_izq += 1

    def contar_tick_der(self):
        if self.enc_der_b.is_active:
            self.ticks_der += 1
        else:
            self.ticks_der -= 1

    def cmd_vel_callback(self, msg):
        self.get_logger().info(f"Recibido: Lineal={msg.linear.x}, Angular={msg.angular.z}")
        # Cinemática inversa: De velocidad lineal/angular a velocidad de cada rueda
        v = msg.linear.x
        w = msg.angular.z
        
        v_izq = v + (w * self.L / 2.0)
        v_der = v - (w * self.L / 2.0)
        
        # Mapear de m/s a PWM (-1.0 a 1.0 para gpiozero)
        
        pwm_izq = max(-1.0, min(1.0, v_izq / 0.5))
        pwm_der = max(-1.0, min(1.0, v_der / 0.5))
        
        self.motor_izq.value = pwm_izq
        self.motor_der.value = pwm_der

    def publicar_odometria(self):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        
        # Calcular distancia recorrida por cada rueda
        dist_izq = 2.0 * math.pi * self.R * (self.ticks_izq / self.TICKS_POR_VUELTA)
        dist_der = 2.0 * math.pi * self.R * (self.ticks_der / self.TICKS_POR_VUELTA)
        
        # Reseteamos los ticks para el siguiente ciclo
        self.ticks_izq = 0
        self.ticks_der = 0
        
        # Cinemática directa
        d_centro = (dist_izq + dist_der) / 2.0
        d_theta = (dist_der - dist_izq) / self.L
        
        # Actualizar posición global
        self.x += d_centro * math.cos(self.theta + (d_theta / 2.0))
        self.y += d_centro * math.sin(self.theta + (d_theta / 2.0))
        self.theta += d_theta
        
        # Crear y llenar el mensaje de Odometría
        odom = Odometry()
        odom.header.stamp = current_time.to_msg()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        # En 2D, la rotación se maneja en el eje Z (yaw) usando cuaterniones
        odom.pose.pose.orientation.z = math.sin(self.theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.theta / 2.0)
        
        # Velocidades instantáneas
        if dt > 0:
            odom.twist.twist.linear.x = d_centro / dt
            odom.twist.twist.angular.z = d_theta / dt
            
        self.odom_pub.publish(odom)
        self.last_time = current_time

def main(args=None):
    rclpy.init(args=args)
    nodo = RobotCore()
    rclpy.spin(nodo)
    nodo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
