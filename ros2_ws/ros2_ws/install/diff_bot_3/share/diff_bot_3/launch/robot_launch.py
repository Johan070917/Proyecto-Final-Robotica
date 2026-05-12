from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. El Cerebro (Intacto)
        Node(
            package='diff_bot_3',
            executable='robot_core',
            name='robot_core_node'
        ),
        
        # 2. El Piloto Automático (Intacto, como pediste)
        Node(
            package='diff_bot_3',
            executable='robot_player',
            name='robot_player_node'
        ),

        # --- LA MEJORA DEL CONTROL BLUETOOTH ---
        # 3. Lector del control físico
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node'
        ),
        
        # 4. Traductor de botones a velocidades
        # 4. Traductor de botones a velocidades (MODO DIOS: Sin embrague)
        Node(
            package='teleop_twist_joy',
            executable='teleop_node',
            name='teleop_twist_joy_node',
            parameters=[{
                'require_enable_button': False, # ¡MAGIA! Apaga el botón de seguridad
                'axis_linear.x': 1,             # Eje 1: Palanca Izquierda (Adelante/Atrás)
                'axis_angular.yaw': 0,          # Eje 0: Palanca Izquierda (Giro)
                'scale_linear.x': 0.5,          # Velocidad máxima lineal (m/s)
                'scale_angular.yaw': 1.0        # Velocidad máxima angular (rad/s)
            }]
        )
    ])

        