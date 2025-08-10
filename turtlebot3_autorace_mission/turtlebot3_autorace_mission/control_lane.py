#!/usr/bin/env python3
#
# Copyright 2018 ROBOTIS CO., LTD.
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
# Author: Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
# from std_msgs.msg import String
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64
from std_msgs.msg import String, UInt8



class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')
        self.label = self.create_subscription(
            String, 
            '/traffic_light_state', 
            self.callback_label, 
            10
        )
        self.label = "NONE"

        self.sub_human = self.create_subscription(
            String,
            '/control/human',
            self.callback_human,
            1
        )
        self.human = "NONE"
        # self.msg_light = self.create_subscription(
        #     String,
        #     '/control/label',
        #     self.callback_light,
        #     1
        # )
        self.stop_line_detected = False
        
        # self.src_light = self.create_client(
        #     YOLO,
        #     '/control/label'
        # )
        self.sub_lane = self.create_subscription(
            Float64,
            '/control/lane',
            self.callback_follow_lane,
            1
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.callback_get_max_vel,
            1
        )
        self.sub_avoid_cmd = self.create_subscription(
            Twist,
            '/avoid_control',
            self.callback_avoid_cmd,
            1
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.callback_avoid_active,
            1
        )

        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            1
        )

        # PD control related variables
        self.last_error = 0
        self.MAX_VEL = 0.1

        # Avoidance mode related variables
        self.avoid_active = False
        self.avoid_twist = Twist()

        self.lane_state = 0

        self.sub_lane_state = self.create_subscription(
            UInt8,
            '/detect/lane_state',
            self.callback_lane_state,
            1
        )

        self.changing_lane = False
        self.bias = 0
        self.last_error = 0
        self.dashed_detected = False
        self.prev_center = 500

        self.sub_dashed = self.create_subscription(
            Bool,
            '/detect/dashed_line',
            self.callback_dashed_line,
            1
        )

    def callback_dashed_line(self, msg):
        if msg.data and not self.changing_lane:
            self.get_logger().info("🔄 점선 감지됨! 차선 변경 시작")
            self.changing_lane = True
            self.bias = -150


    def callback_lane_state(self, msg):
        self.lane_state = msg.data

    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    # def callback_light(self, light):
    #     self.label = light.data

    def callback_human(self, human):
        self.human = human.data
        if self.human == "Stop":
            self.get_logger().info("Human detected! Stop.")
        elif self.human == "Slow":
            self.get_logger().info("Human detected! Slow.")


    def callback_label(self, msg):
        self.label = msg.data
        if self.label == "red_light":
            self.get_logger().info("Red light detected! Stopping.")

    def callback_follow_lane(self, desired_center):
        """
        Receive lane center data to generate lane following control commands.

        If avoidance mode is enabled, lane following control is ignored.
        """
        if self.avoid_active:
            return

        center = desired_center.data

        # === 차선 변경 중일 경우: 일정 시간동안 bias 유지 ===
        if self.dashed_detected:
            self.get_logger().info("점선 감지 → 차선 변경 시작")
            self.changing_lane = True
            self.dashed_detected = False


        # === 중심값 결정 ===
        if self.lane_state == 0 and self.changing_lane:
            # 이전 중심값 사용
            # center = self.prev_center + self.bias
            twist = Twist()
            twist.linear.x = 0.05
            twist.angular.z = 0.
            self.pub_cmd_vel.publish(twist)
            self.get_logger().warn("lane_state == 0: 직진")
            return

        if self.lane_state == 1 and self.changing_lane:
            self.get_logger().info("lane_state == 1: 차선 변경 완료")
            self.changing_lane = False
            self.bias = 0

        center = desired_center.data + self.bias
        error = center - 500

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error

        twist = Twist()


        # Linear velocity: adjust speed based on error (maximum 0.05 limit)
        if ("RED" == self.label and self.stop_line_detected) or "Stop" == self.human:
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            self.pub_cmd_vel.publish(twist)
        elif "yellow_ligth" == self.label or "Slow" == self.human:
            twist.linear.x = (min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05))/2
        else:
            twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05)
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.pub_cmd_vel.publish(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.pub_cmd_vel.publish(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
