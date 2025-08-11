from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='turtlebot3_autorace_mission',
            executable='control_lane',
            name='control_lane',
            output='screen',
            remappings=[
                ('cmd_vel', '/auto_cmd_vel'),
                ('/cmd_vel', '/auto_cmd_vel'),
                ('avoid_control', '/auto_cmd_vel'),
                ('/avoid_control', '/auto_cmd_vel'),
                ('control/cmd_vel', '/auto_cmd_vel'),
                ('/control/cmd_vel', '/auto_cmd_vel'),
            ],
        ),
    ])
