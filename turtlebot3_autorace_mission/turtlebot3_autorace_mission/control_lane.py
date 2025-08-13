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

from rclpy.qos import (
    QoSProfile,
    QoSHistoryPolicy,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
)

class ControlLane(Node):
    def publish_cmd(self, twist):
        """Publish Twist to /control/cmd_vel."""
        self.pub_cmd_vel.publish(twist)


    def __init__(self):
        super().__init__('control_lane')
        # === Unified control loop ===
        self._pending_twist = None

        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/control/cmd_vel',
            1
        )

        self.control_timer = self.create_timer(0.05, self.control_step)

        self._last_cmd = Twist()
        self._last_cmd_time = self.get_clock().now()
        self.hold_ms = 300  # 0.3초 이내면 마지막 명령 유지

        # === 가속 제어 상태 ===
        self.current_speed = 0.09   # 시작 속도
        self.max_speed     = 0.3   # 최대 속도(원하면 0.25~0.3)
        self.min_speed     = 0.06   # 최소 속도 지정
        self.accel_step    = 0.005  # 콜백당 가속량
        self.decel_step    = 0.005  # 콜백당 감속량(조금 더 크게)
        self.yawrate_ok    = 0.11   # |ωz| 임계(작을수록 직진일 때만 가속)
        self.error_ok_px   = 50.0   # 차선중심 오차 허용 픽셀

        # === Odom 구독 ===
        self.odom_lin_x = 0.0
        self.odom_yaw_z = 0.0
        self.sub_odom = self.create_subscription(
            Odometry, '/odom', self.callback_odom, 10
        )

        self.is_stopped = False

        self.label = "NONE"
        self.label = self.create_subscription(
            String, 
            '/traffic_light_state', 
            self.callback_label, 
            10
        )


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
        

        lane_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub_lane = self.create_subscription(
            Float64,
            '/detect/lane',
            self.callback_follow_lane,
            lane_qos,
        )
        self.sub_max_vel = self.create_subscription(
            Float64,
            '/control/max_vel',
            self.callback_get_max_vel,
            1
        )
        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.callback_avoid_active,
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
        self.publish_cmd(twist)
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

    def callback_avoid_active(self, msg: Bool):
        """회피 모드 on/off 플래그 수신"""
        self.avoid_active = bool(msg.data)
        # 필요하면 로그
        # self.get_logger().info(f"Avoid active: {self.avoid_active}")

    def callback_avoid_cmd(self, twist_msg: Twist):
        self.avoid_twist = twist_msg
        if self.avoid_active:
            self._pending_twist = self.avoid_twist  # 즉시 publish 대신 pending


    def _safe_publish(self, twist: Twist):
        # 멈춤 상태면 무조건 0으로 오버라이드
        if self.is_stopped:
            zero = Twist()
            self.publish_cmd(zero)
            return
        self.publish_cmd(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.publish_cmd(self.avoid_twist)

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
        self.human = human.data  # "Stop" / "Slow" / "Go"(또는 "GO")
        # 사람이 보이면 정지 게이팅, 아니면 해제
        if self.human.lower() == "stop":
            self.person_detected = True
            self.slow_factor = 0.0      # 완전 정지
            self.get_logger().info("Human detected! Stop.")
        elif self.human.lower() == "slow":
            self.person_detected = False
            self.slow_factor = 0.5      # 느리게
            self.get_logger().info("Human detected! Slow.")
        else:  # "go" 또는 그 외는 정상 주행
            self.person_detected = False
            self.slow_factor = 1.0


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
        - 콜백에서는 _pending_twist만 갱신하고, 실제 퍼블리시는 control_step()에서만 수행.
        - 사람/신호/정지선 등 안전 조건이 최우선.
        """
        # 0) 사람 감지: 최우선 정지
        if getattr(self, 'person_detected', False):
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self._pending_twist = stop
            return

        # 1) 신호등 상태 머신(정지/해제) — is_stopped 단일 관리
        label = getattr(self, 'label', 'NONE')
        if self.is_stopped:
            # GREEN이면 해제
            if label == "GREEN":
                self.get_logger().info("🟢 GREEN detected → 주행 재개")
                self.is_stopped = False
            else:
                stop = Twist()
                stop.linear.x = 0.0
                stop.angular.z = 0.0
                self._pending_twist = stop
                return
        else:
            # 정지 조건 진입: RED + 정지선
            if label == "RED" and getattr(self, 'stop_line_state', False):
                self.get_logger().info("🔴 RED + Stop line → 정지")
                self.is_stopped = True
                stop = Twist()
                stop.linear.x = 0.0
                stop.angular.z = 0.0
                self._pending_twist = stop
                return

        # 2) 차선 변경 트리거(점선 + 회피 활성)
        if getattr(self, 'dashed_detected', False) and getattr(self, 'avoid_active', False):
            self.get_logger().info("점선 감지 → 차선 변경 시작")
            if self.dashed_dir == "left":
                self.bias = -150
            elif self.dashed_dir == "right":
                self.bias = 160
            self.changing_lane = True
            self.dashed_detected = False

        # 3) 차선 변경 완료 판정
        if getattr(self, 'changing_lane', False):
            # lane_state == 1 (왼차선 유지) + left 변경 OR lane_state == 3 (오른차선 유지) + right 변경 → 종료
            if (self.lane_state == 1 and self.dashed_dir == "left") or (self.lane_state == 3 and self.dashed_dir == "right"):
                self.changing_lane = False
                self.bias = 0

        # 4) 중심/오차/제어
        center = desired_center.data + getattr(self, 'bias', 0)
        error = center - 500

        Kp = 0.0025
        Kd = 0.007
        angular_cmd = Kp * error + Kd * (error - getattr(self, 'last_error', 0))
        self.last_error = error
        # 한 번만 클리핑
        angular_cmd = float(np.clip(angular_cmd, -2.0, 2.0))

        # 5) 속도 — MAX_VEL 기반 곡선 + 표지판(50km) 배율
        # lane_state: 1/3만 "양호"로 간주 (2=both lanes는 제외)
        good_lane = self.lane_state in (1, 3) and abs(error) < getattr(self, 'error_ok_px', 40)
        base_speed = min(self.MAX_VEL * (max(1 - abs(error) / 500, 0) ** 2.2), 0.3)
        if not good_lane:
            base_speed = max(base_speed, getattr(self, 'min_speed', 0.02))  # 바닥 속도 보장

        if getattr(self, 'sign', '') == "km_50":
            base_speed *= 1.5  # 기존 코드의 *5는 과함. 필요시 파라미터로.

        # 차선 미인식(0)일 땐 회전 정지(직진 유지)
        if self.lane_state == 0:
            angular_cmd = 0.0

        # 6) 최종 Twist — 부호는 한 번만 반영
        twist = Twist()
        twist.linear.x = base_speed
        twist.angular.z = -angular_cmd   # 카메라 좌표/오차 정의에 맞춰 한 번만 반전

        # callback_follow_lane() 끝에서 pending 세팅하는 부분에 추가
        self._pending_twist = twist
        self._last_cmd = twist
        self._last_cmd_time = self.get_clock().now()



    def control_step(self):
        """Single output loop with priority arbitration."""
        # 1) Person detected => full stop (최우선)
        if getattr(self, 'person_detected', False) or self.human.lower() == "stop":
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.publish_cmd(stop)
            return

        # 2) RED + stop line => stop
        if getattr(self, 'label', 'NONE') == 'RED' and getattr(self, 'stop_line_state', False):
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.publish_cmd(stop)
            return

        # 3) is_stopped 상태 유지
        if getattr(self, 'is_stopped', False):
            stop = Twist()
            stop.linear.x = 0.0
            stop.angular.z = 0.0
            self.publish_cmd(stop)
            return

        # 4) Pending twist 있으면 그것만 퍼블리시
        if getattr(self, '_pending_twist', None) is not None:
            tw = self._pending_twist
            # 사람 SLOW가 걸려 있으면 선형속도만 배율
            try:
                tw.linear.x *= getattr(self, 'slow_factor', 1.0)
            except Exception:
                pass
            self.publish_cmd(tw)
            self._pending_twist = None
            return

        
        # control_step() 퍼블리시 로직 수정
        if self._pending_twist is not None:
            self.publish_cmd(self._pending_twist)
            self._last_cmd = self._pending_twist
            self._last_cmd_time = self.get_clock().now()
            self._pending_twist = None
            return

        age_ms = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e6
        if age_ms < self.hold_ms:
            self.publish_cmd(self._last_cmd)
            return

        zero = Twist()
        self.publish_cmd(zero)

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