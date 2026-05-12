"""
robot_launch.py  -  Lanza todo el sistema en la Raspberry Pi 5.

  * robot_core              (PID + odometria; habla con ESP32_MOTORS)
  * cmd_vel_mux             (prioriza teleop > auto > player)
  * robot_player            (reproductor de trayectorias en lazo cerrado)
  * vision_node             (deteccion de cubos R/G/B + MJPEG :8080)
  * forklift_manager        (FSM autonomo del montacargas)
  * esp32_bridge_motors     (USB <-> ESP32_MOTORS)  /motors_cmd, /motors_status
  * esp32_bridge_lift       (USB <-> ESP32_LIFT)    /lift_cmd,   /lift_status
  * joy_node                (joystick bluetooth)
  * teleop_twist_joy        (Joy -> /cmd_vel_teleop)

La interfaz grafica (robot_interface) se lanza aparte porque requiere X.
"""

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node


def generate_launch_description():

    # Por defecto, "auto": cada bridge escanea /dev/ttyUSB* y /dev/ttyACM*
    # y se queda con la ESP32 que responde WHOAMI con su ID esperado
    # (MOTORS o LIFT). Si por algun motivo la autodeteccion falla, se
    # puede forzar un puerto especifico:
    #   ros2 launch diff_bot_3 robot_launch.py motors_port:=/dev/ttyUSB3
    motors_port = DeclareLaunchArgument(
        'motors_port', default_value='auto',
        description='Puerto serie de la ESP32 de motores ("auto" = autodetectar)')

    lift_port = DeclareLaunchArgument(
        'lift_port', default_value='auto',
        description='Puerto serie de la ESP32 del montacargas ("auto" = autodetectar)')

    return LaunchDescription([
        motors_port,
        lift_port,

        Node(package='diff_bot_3', executable='robot_core',
             name='robot_core'),

        Node(package='diff_bot_3', executable='cmd_vel_mux',
             name='cmd_vel_mux'),

        Node(package='diff_bot_3', executable='robot_player',
             name='robot_player'),

        Node(package='diff_bot_3', executable='vision_node',
             name='vision_node'),

        Node(package='diff_bot_3', executable='forklift_manager',
             name='forklift_manager'),

        # ---- ESP32 motores ----
        Node(package='diff_bot_3', executable='esp32_bridge',
             name='esp32_bridge_motors',
             parameters=[{
                 'port':         LaunchConfiguration('motors_port'),
                 'expected_id':  'MOTORS',
                 'baud':         115200,
                 'cmd_topic':    '/motors_cmd',
                 'status_topic': '/motors_status',
             }]),

        # ---- ESP32 montacargas ----
        Node(package='diff_bot_3', executable='esp32_bridge',
             name='esp32_bridge_lift',
             parameters=[{
                 'port':         LaunchConfiguration('lift_port'),
                 'expected_id':  'LIFT',
                 'baud':         115200,
                 'cmd_topic':    '/lift_cmd',
                 'status_topic': '/lift_status',
             }]),

        # ---- Joystick bluetooth -> /cmd_vel_teleop ----
        Node(package='joy', executable='joy_node', name='joy_node'),

        Node(package='teleop_twist_joy',
             executable='teleop_node',
             name='teleop_twist_joy',
             parameters=[{
                 'require_enable_button': False,
                 'axis_linear.x':   1,
                 'axis_angular.yaw': 0,
                 'scale_linear.x':   0.15,
                 'scale_angular.yaw': 1.00,
             }],
             remappings=[('/cmd_vel', '/cmd_vel_teleop')]),
    ])
