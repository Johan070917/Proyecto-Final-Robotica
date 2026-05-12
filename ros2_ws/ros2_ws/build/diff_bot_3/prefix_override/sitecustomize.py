import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/johan/Taller-2-Robotica/ros2_ws/install/diff_bot_3'
