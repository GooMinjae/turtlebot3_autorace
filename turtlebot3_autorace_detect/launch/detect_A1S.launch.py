import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot3_autorace_detect')

    calibration_mode_arg = DeclareLaunchArgument(
        'calibration_mode',
        default_value='False',
        description='Mode type [calibration, action]'
    )
    calibration_mode = LaunchConfiguration('calibration_mode')

    detect_param = os.path.join(
        pkg_share,
        'param',
        'lane',
        'lane.yaml'
        )
    param_file = os.path.join(pkg_share, 'param', 'traffic_light', 'traffic_light.yaml')

    detect_lane_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_lane',
        name='detect_lane',
        output='screen',
        parameters=[
            {'is_detection_calibration_mode': calibration_mode},
            detect_param
        ],
        remappings=[
            ('/detect/image_input', '/camera/image_projected'),
            ('/detect/image_input/compressed', '/camera/image_projected/compressed'),
            ('/detect/image_output', '/detect/image_lane'),
            ('/detect/image_output/compressed', '/detect/image_lane/compressed'),
            ('/detect/image_output_sub1', '/detect/image_white_lane_marker'),
            ('/detect/image_output_sub1/compressed', '/detect/image_white_lane_marker/compressed'),
            ('/detect/image_output_sub2', '/detect/image_yellow_lane_marker'),
            ('/detect/image_output_sub2/compressed', '/detect/image_yellow_lane_marker/compressed')
        ]
    )
    detect_traffic_light_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_traffic_light',
        name='detect_traffic_light',
        output='screen',
        parameters=[
            param_file,
            {'is_calibration_mode': calibration_mode}
        ],
        remappings=[
            ('/detect/image_input', '/camera/image_compensated'),
            ('/detect/image_input/compressed', '/camera/image_compensated/compressed'),
            ('/detect/image_output', '/detect/image_traffic_light'),
            ('/detect/image_output/compressed', '/detect/image_traffic_light/compressed'),
            ('/detect/image_output_sub1', '/detect/image_red_light'),
            ('/detect/image_output_sub1/compressed', '/detect/image_red_light/compressed'),
            ('/detect/image_output_sub2', '/detect/image_yellow_light'),
            ('/detect/image_output_sub2/compressed', '/detect/image_yellow_light/compressed'),
            ('/detect/image_output_sub3', '/detect/image_green_light'),
            ('/detect/image_output_sub3/compressed', '/detect/image_green_light/compressed'),
        ],
    )
    detect_human_node = Node(
        package='turtlebot3_autorace_detect',
        executable='person_detector',
        name='person_detector',
        output='screen',
        remappings=[
            ('/detect/human', '/control/human'),
        ],
    )
    return LaunchDescription([
        calibration_mode_arg,
        detect_lane_node,
        detect_traffic_light_node,
        detect_human_node
    ])
