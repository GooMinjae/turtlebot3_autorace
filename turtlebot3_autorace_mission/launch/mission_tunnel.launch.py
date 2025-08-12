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







    person_node = Node(
            package='turtlebot3_autorace_detect',
            executable='person_detector',
            name='person_detector',
            output='screen',
            remappings=[
            ]
    )

    return LaunchDescription([
        # avoid_object_node,
        # detect_lane_node,
        # detect_traffic_light_node,
        person_node,
        # detect_sign_node,
        # detect_speed_node,
        # control_node,

    ])
