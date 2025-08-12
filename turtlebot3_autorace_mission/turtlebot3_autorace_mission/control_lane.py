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
from nav_msgs.msg import Odometry
import numpy as np
import time


class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')
        # === 가속 제어 상태 ===
        self.current_speed = 0.07   # 시작 속도
        self.max_speed     = 0.2   # 최대 속도(원하면 0.25~0.3)
        self.min_speed     = 0.05   # 🚀 최소 속도 지정
        self.accel_step    = 0.002  # 콜백당 가속량
        self.decel_step    = 0.01  # 콜백당 감속량(조금 더 크게)
        self.yawrate_ok    = 0.11   # |ωz| 임계(작을수록 직진일 때만 가속)
        self.error_ok_px   = 50.0   # 차선중심 오차 허용 픽셀

        # === Odom 구독 ===
        self.odom_lin_x = 0.0
        self.odom_yaw_z = 0.0
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.callback_odom, 10
        )

        self.is_stopped = False

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

        # 상태 변수
        self.stop_line_state = False

        # Subscriber 객체 (이건 변수명 다른 걸로)
        self.sub_stop_line = self.create_subscription(
            Bool,
            '/detect/stop_line',
            self.callback_stop_line,
            1
        )
        
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
        self.sign="NONE"
        self.sub_sign = self.create_subscription(
            String,
            '/detect/sign',
            self.callback_sign,
            1
        )

        # self.inter_sign="None"
        # self.sub_inter_sign = self.create_subscription(
        #     String,
        #     '/detect/inter_sign',
        #     self.callback_inter_sign,
        #     1
        # )


        # PD control related variables
        self.last_error = 0
        self.MAX_VEL = 0.1
        self.wait_for_green = False

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
            String,
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
            1
        )
        self.values = []

        self.pub_reset_dashed = self.create_publisher(Bool, '/detect/reset_dashed', 1)


    def _publish_stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub_cmd_vel.publish(twist)
    # -------------------------------------------------------------------------
    # 추가: /person_detected 콜백
    #  - 단순히 내부 상태 변수(self.person_detected) 갱신
    # -----------------------------ㅊ--------------------------------------------
    def cb_person_flag(self, msg: Bool):
        self.person_detected = bool(msg.data)
        # 디버깅 원하면 아래 로그를 잠깐 켜도 좋음
        # self.get_logger().info(f'[person_detected] {self.person_detected}')
    # -----------------------------ㅊ--------------------------------------------

    def callback_odom(self, msg: Odometry):
        self.odom_lin_x = float(msg.twist.twist.linear.x)
        self.odom_yaw_z = float(msg.twist.twist.angular.z)

    def _safe_publish(self, twist: Twist):
        # 멈춤 상태면 무조건 0으로 오버라이드
        if self.is_stopped:
            zero = Twist()
            self.pub_cmd_vel.publish(zero)
            return
        self.pub_cmd_vel.publish(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.pub_cmd_vel.publish(self.avoid_twist)

    def callback_dashed_line(self, msg):
        if (msg.data in ["right", "left"]) and not self.changing_lane:
            self.get_logger().info("🔄 점선 감지됨! 차선 변경 시작")
            self.changing_lane = True
            self.dashed_detected = True
            self.dashed_dir = msg.data
            # self.bias = -150


    def callback_lane_state(self, msg):
        self.lane_state = msg.data

    def callback_get_max_vel(self, max_vel_msg):
        self.MAX_VEL = max_vel_msg.data

    # def callback_light(self, light):
    #     self.label = light.data
    def callback_stop_line(self, msg):
        self.stop_line_state = msg.data
        if self.stop_line_state:
            self.get_logger().info("Stop line detected! Stopping.")
        



    def callback_human(self, human):
        self.human = human.data
        if self.human == "Stop":
            self.get_logger().info("Human detected! Stop.")
        elif self.human == "Slow":
            self.get_logger().info("Human detected! Slow.")

    def callback_sign(self,msg):
        self.sign = msg.data

    # def callback_inter_sign(self,msg):
    #     self.inter_sign = msg.data


    def callback_label(self, msg):
        self.label = msg.data
        if self.label == "RED":
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

        # center = desired_center.data
        # center = desired_center.data + self.bias

        # 2) 속도 0 이후 '신호 대기' 상태 처리:
        #    빨간불이면 정지 유지, 빨간불 아니면 자동 재출발
        if self.wait_for_green:
            if self.label != "RED":
                # 자동 재출발: 최소 속도로 시작
                self.wait_for_green = False
                self.current_speed = max(self.current_speed, self.min_speed)
            else:
                self._publish_stop()
                return

        # 차선 변경 로직
        # === 차선 변경 중일 경우: 일정 시간동안 bias 유지 ===
        # if self.dashed_detected:
        if self.dashed_detected and self.avoid_active:
            self.get_logger().info("점선 감지 → 차선 변경 시작")
            if self.dashed_dir == "left":
                self.bias = -150
            elif self.dashed_dir == "right":
                self.bias = 160
            self.changing_lane = True
            self.dashed_detected = False


        # # === 중심값 결정 ===
        # if self.lane_state == 0 and self.changing_lane:
        #     twist = Twist()
        #     twist.linear.x = 0.06
        #     twist.angular.z = 0.
        #     self.pub_cmd_vel.publish(twist)
        #     self.get_logger().warn("lane_state == 0: 직진")
        #     return

        if self.changing_lane:
            if (self.lane_state == 1 and self.dashed_dir == "left") or (self.lane_state == 3 and self.dashed_dir == "right"):
                # self.get_logger().info("lane_state == 1: 차선 변경 완료")
                self.changing_lane = False
                self.bias = 0
                # self.pub_reset_dashed.publish(Bool(data=True))

        # self.get_logger().info(f"{self.bias}")

        center = desired_center.data + self.bias
        error = center - 500

        Kp = 0.0025
        Kd = 0.007

        angular_z = Kp * error + Kd * (error - self.last_error)
        self.last_error = error
        angular_z = float(np.clip(angular_z, -2.0, 2.0))

        # === 험한 지형을 위한 안정적인 속도 제어 로직 ===
        # 차선 상태가 양호하고 (1, 3), 차선 오차가 작을 때만 가속
        if self.lane_state in (1, 2, 3) and abs(error) < self.error_ok_px:
            # 차선이 잘 보이고, 오차가 작을 때만 가속
            self.current_speed = min(self.current_speed + self.accel_step, self.max_speed)
            linear_x = self.current_speed
        else:
            # 차선이 불안정하거나 오차가 클 때
            self.current_speed = self.min_speed
            linear_x = self.min_speed
            
            # 차선이 인식되지 않을 땐 회전도 멈춤 (직진)
            if self.lane_state == 0:
                angular_z = 0.0

        # 신호/사람 Slow는 최종 게이팅
        if self.label.startswith("YELLOW") or self.human == "SLOW":
            linear_x *= 0.5
        
        # 최종 Twist 메시지 생성
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = -angular_z


        # callback_follow_lane 안
        # 🟢 GREEN → 정지 상태 해제
        if self.label == "GREEN":
            if self.is_stopped:
                self.get_logger().info("🟢 GREEN detected → 주행 재개")
            self.is_stopped = False

        # 🔴 RED + 정지선 감지 또는 사람 Stop → 정지 상태 진입 (한 번만 세팅)
        if not self.is_stopped and (self.label == "RED" and self.stop_line_state == True):
            self.get_logger().info("🔴 RED/Human Stop → 정지 상태 진입")
            self.is_stopped = True

        # 🚫 정지 상태면 무조건 멈춤 (stop_line_detected가 False여도 유지)
        if self.is_stopped:
            twist = Twist()
            twist.linear.x = 0.0
            twist.linear.y = 0.0
            self.pub_cmd_vel.publish(twist)
            return
        if self.sign == "km_50":
            twist.linear.x = (min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05)) *5
        # elif "intersection" == self.inter_sign:
        #     twist.linear.x = (min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05))/2
        else:
            twist.linear.x = (min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.05)) * 2
        twist.angular.z = -max(angular_z, -2.0) if angular_z < 0 else -min(angular_z, 2.0)
        self.pub_cmd_vel.publish(twist)

        # ---------------------------------------------------------------------

        # self.pub_cmd_vel.publish(twist)

    # def callback_avoid_cmd(self, twist_msg):
    #     self.avoid_twist = twist_msg
    #     # ---------------------------------------------------------------------
    #     # 추가: 회피 모드 중이라도 사람 감지면 무조건 정지
    #     #  - 회피보다 사람 안전을 최우선으로 둠
    #     # ---------------------------------------------------------------------
    #     if self.person_detected:
    #         stop = Twist()
    #         self.pub_cmd_vel.publish(stop)
    #         return
    #     # ---------------------------------------------------------------------

    #     if self.avoid_active:
    #         self.pub_cmd_vel.publish(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        # if self.avoid_active:
        #     self.get_logger().info('Avoidance mode activated.')
        # else:
        #     self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

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
