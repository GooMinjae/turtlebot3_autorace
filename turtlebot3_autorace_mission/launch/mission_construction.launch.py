#!/usr/bin/env python3
#
# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: ChanHyeong Lee

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('turtlebot3_autorace_detect')

    detect_param = os.path.join(
        pkg_share,
        'param',
        'lane',
        'lane.yaml'
    )

    avoid_object_node = Node(
        package='turtlebot3_autorace_mission',
        executable='avoid_construction',
        name='avoid_construction',
        output='screen'
    )

    detect_lane_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_lane',
        name='detect_lane',
        output='screen',
        parameters=[
            {'is_detection_calibration_mode': True},
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
    param_file = os.path.join(pkg_share, 'param', 'traffic_light', 'traffic_light.yaml')

    detect_traffic_light_node = Node(
        package='turtlebot3_autorace_detect',
        executable='detect_traffic_light',
        name='detect_traffic_light',
        output='screen',
        parameters=[
            param_file,
            {'is_calibration_mode': True}
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

    control_node = Node(
            package='turtlebot3_autorace_mission',
            executable='control_lane',
            name='control_lane',
            output='screen',
            remappings=[
                ('/control/lane', '/detect/lane'),
                ('/control/cmd_vel', '/cmd_vel')
            ]
        )

    person_node = Node(
            package='turtlebot3_autorace_detect',
            executable='person_detector',
            name='person_detector',
            output='screen',
            remappings=[
            ]
    )

    return LaunchDescription([
        avoid_object_node,
        detect_lane_node,
        detect_traffic_light_node,
        person_node,
        control_node,

    ])
