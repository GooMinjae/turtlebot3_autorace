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

        self.stop_line_detected = self.create_subscription(
             Bool,
             '/detect/stop_line',
             self.callback_stop_line,
             1
        )
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
        # self.sign="NONE"
        # self.sub_sign = self.create_subscription(
        #     String,
        #     '/detect/sign',
        #     self.callback_sign,
        #     1
        # )

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

        self.sub_dashed = self.create_subscription(
            Bool,
            '/detect/dashed_line',
            self.callback_dashed_line,
            1
        )
        # ---------------------------------------------------------------------
        # 추가: 사람 감지 플래그(/person_detected) 구독 + 상태 변수
        #  - person_detector.py가 보내는 Bool(True=사람 영향 있음)을 받아서
        #    최종 퍼블리시 직전에 0속도 게이팅에 활용
        # ---------------------------------------------------------------------
        self.person_detected = False
        self.sub_person_flag = self.create_subscription(
            Bool,
            '/person_detected',
            self.cb_person_flag,
            10
        )
    # -------------------------------------------------------------------------
    # 추가: /person_detected 콜백
    #  - 단순히 내부 상태 변수(self.person_detected) 갱신
    # -----------------------------ㅊ--------------------------------------------
    def cb_person_flag(self, msg: Bool):
        self.person_detected = bool(msg.data)
        # 디버깅 원하면 아래 로그를 잠깐 켜도 좋음
        # self.get_logger().info(f'[person_detected] {self.person_detected}')
    # -----------------------------ㅊ--------------------------------------------

    def callback_dashed_line(self, msg):
        if msg.data and not self.changing_lane:
            self.get_logger().info("🔄 점선 감지됨! 차선 변경 시작")
            self.changing_lane = True
            self.dashed_detected = True
            # self.bias = -150


    def callback_lane_state(self, msg):
        self.lane_state = msg.data

    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    # def callback_light(self, light):
    #     self.label = light.data
    def callback_stop_line(self, msg):
        self.stop_line_detected = msg.data
        if self.stop_line_detected:
            self.get_logger().info("Stop line detected! Stopping.")

    def callback_human(self, human):
        self.human = human.data
        if self.human == "Stop":
            self.get_logger().info("Human detected! Stop.")
        elif self.human == "Slow":
            self.get_logger().info("Human detected! Slow.")
    # def callback_sign(self, sign):
    #     self.sign = sign.data

    def callback_label(self, msg):
        self.label = msg.data
        if self.label == "red_light":
            self.get_logger().info("Red light detected! Stopping.")

    def callback_follow_lane(self, desired_center):
        """
        Receive lane center data to generate lane following control commands.

        If avoidance mode is enabled, lane following control is ignored.

        """
        # ▶ 우선 조건: 양쪽 차선이면 정지
        # if self.lane_state == 2:
        #     twist = Twist()
        #     twist.linear.x = 0.0
        #     twist.angular.z = 0.0
        #     self.pub_cmd_vel.publish(twist)
        #     self.get_logger().warn("lane_state == 2 (both lanes): STOP.")
        #     return
        # if self.avoid_active:
        #     return
        # ---------------------------------------------------------------------
        # 추가: 최우선 안전 규칙 — 사람 감지 시 즉시 정지 후 리턴
        #  - 어떤 상황(회피/신호등/차선변경)보다도 우선
        # ---------------------------------------------------------------------
        if self.person_detected:
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.pub_cmd_vel.publish(stop)
            return
        # ---------------------------------------------------------------------


        center = desired_center.data

        # === 차선 변경 중일 경우: 일정 시간동안 bias 유지 ===
        # if self.dashed_detected:
        if self.dashed_detected and self.avoid_active:
            self.get_logger().info("점선 감지 → 차선 변경 시작")
            self.bias = -150
            self.changing_lane = True
            self.dashed_detected = False


        # === 중심값 결정 ===
        if self.lane_state == 0 and self.changing_lane:
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

        self.get_logger().info(f"{self.bias}")

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
        elif "YELLOW" == self.label or "Slow" == self.human:
            twist.linear.x = (min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05))/2
        else:
            twist.linear.x = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05)
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        # if self.sign == "left":
        #     twist.angular.z = 0.3
        self.pub_cmd_vel.publish(twist)

        # ---------------------------------------------------------------------
        # 추가: 퍼블리시 직전 한 번 더 게이팅
        #  - 콜백 처리 중간에 /person_detected가 True로 바뀌는 레이스 컨디션 방지
        # ---------------------------------------------------------------------
        if self.person_detected:
            stop = Twist()
            self.pub_cmd_vel.publish(stop)
            return
        # ---------------------------------------------------------------------

        self.pub_cmd_vel.publish(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg
        # ---------------------------------------------------------------------
        # 추가: 회피 모드 중이라도 사람 감지면 무조건 정지
        #  - 회피보다 사람 안전을 최우선으로 둠
        # ---------------------------------------------------------------------
        if self.person_detected:
            stop = Twist()
            self.pub_cmd_vel.publish(stop)
            return
        # ---------------------------------------------------------------------

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
