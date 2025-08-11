# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist
# from cv_bridge import CvBridge
# import cv2
# import numpy as np
# import time

# class PersonDetector(Node):
#     def __init__(self):
#         super().__init__('person_detector')

#         # 카메라 이미지 토픽 구독
#         self.subscription = self.create_subscription(
#             Image,
#             '/camera/image_compensated',  # 필요 시 다른 토픽으로 변경 가능
#             self.image_callback,
#             10
#         )

#         # 속도 명령 퍼블리셔
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         # OpenCV 브리지
#         self.bridge = CvBridge()

#     def image_callback(self, msg):
#         try:
#             # ROS 이미지 → OpenCV 이미지
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#             # BGR → HSV 변환
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#             # 핑크색 HSV 범위 설정
#             lower_pink = np.array([140, 30, 80])   # 색상 조정됨
#             upper_pink = np.array([175, 255, 255])
#             mask = cv2.inRange(hsv, lower_pink, upper_pink)

#             # 핑크색 영역 픽셀 수
#             pink_area = cv2.countNonZero(mask)

#             # 로그 출력
#             self.get_logger().info(f"핑크색 영역: {pink_area}")

#             # 조건에 따라 정지 / 감속 / 주행
#             if pink_area >= 1000:
#                 self.get_logger().info("▶ 사람 감지됨 → 차량 정지")
#                 for _ in range(10):
#                     self.publish_cmd_vel(0.0)
#                     time.sleep(0.05)

#             elif pink_area >= 700:
#                 self.get_logger().info("⚠ 사람 근처 감지됨 → 감속 주행")
#                 self.publish_cmd_vel(0.1)

#             else:
#                 self.publish_cmd_vel(0.2)

#             # 디버깅용 마스크 시각화
#             cv2.imshow("Pink Mask", mask)
#             cv2.waitKey(1)

#         except Exception as e:
#             self.get_logger().error(f"이미지 콜백 오류: {e}")

#     def publish_cmd_vel(self, linear_x):
#         msg = Twist()
#         msg.linear.x = linear_x
#         msg.angular.z = 0.0
#         self.cmd_vel_pub.publish(msg)

# def main(args=None):
#     rclpy.init(args=args)
#     node = PersonDetector()
#     rclpy.spin(node)
#     node.destroy_node()
#     cv2.destroyAllWindows()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
#!/usr/bin/env python3
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        # ===== 설정값 =====
        # 입력 이미지 토픽
        self.topic_image = '/camera/image_compensated'

        # 속도
        self.v_normal = 0.20   # (참고값: MUX 우선순위용, 여기선 발행 안 함)
        self.v_slow   = 0.10

        # 핑크 HSV 범위 (필요시 튜닝)
        self.lower_pink = np.array([140, 30, 80])
        self.upper_pink = np.array([175, 255, 255])

        # 면적 임계값 / 히스테리시스
        self.area_slow    = 700     # SLOW 진입
        self.area_stop    = 1000    # STOP 진입
        self.area_resume  = 500     # GO로 복귀 기준(낮음)
        self.n_frames_trigger = 3   # 진입 연속 프레임
        self.n_frames_release = 3   # 해제 연속 프레임

        # ROI: 상체 검출에 유리한 중간 띠
        # 'middle' | 'bottom' | 'full' | 'custom'
        self.roi_mode = 'custom'
        self.roi_y_start_ratio = 0.35  # custom용
        self.roi_y_end_ratio   = 0.75

        # ===================

        self.bridge = CvBridge()
        self.state = 'GO'  # 'GO' | 'SLOW' | 'STOP'
        self.trigger_cnt = 0
        self.release_cnt = 0

        # 구독 / 퍼블리시
        self.sub = self.create_subscription(Image, self.topic_image, self.image_cb, 10)
        # ⚠️ 중요: 사람 감지 전용 채널로 발행
        self.pub_person = self.create_publisher(Twist, '/person_cmd_vel', 10)

        self.get_logger().info(f'person_detector started. image={self.topic_image}, ROI={self.roi_mode}')

    # ---------- ROI 계산 ----------
    def get_roi(self, frame):
        h, w = frame.shape[:2]
        if self.roi_mode == 'full':
            y0, y1 = 0, h
        elif self.roi_mode == 'bottom':
            y0, y1 = int(h * 0.60), h       # 하단 40%
        elif self.roi_mode == 'middle':
            y0, y1 = int(h * 0.35), int(h * 0.75)
        else:  # custom
            r0 = max(0.0, min(self.roi_y_start_ratio, 0.95))
            r1 = max(r0 + 0.05, min(self.roi_y_end_ratio, 1.0))
            y0, y1 = int(h * r0), int(h * r1)

        y0 = max(0, min(y0, h-1))
        y1 = max(y0+1, min(y1, h))
        roi = frame[y0:y1, :]
        return roi, (y0, y1, w, h)

    # ---------- 메인 콜백 ----------
    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            roi, (y0, y1, w, h) = self.get_roi(frame)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)

            # (선택) 노이즈 처리
            # kernel = np.ones((3,3), np.uint8)
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            # mask = cv2.dilate(mask, kernel, iterations=1)

            pink_area = int(cv2.countNonZero(mask))
            self.get_logger().info(f'핑크 영역(ROI): {pink_area} | state={self.state}')

            # --- 상태 결정(히스테리시스 + 디바운스) ---
            next_state = self.state
            if self.state != 'STOP' and pink_area >= self.area_stop:
                self.trigger_cnt += 1; self.release_cnt = 0
                if self.trigger_cnt >= self.n_frames_trigger:
                    next_state = 'STOP'
            elif self.state == 'STOP' and pink_area <= self.area_resume:
                self.release_cnt += 1; self.trigger_cnt = 0
                if self.release_cnt >= self.n_frames_release:
                    next_state = 'GO'
            elif self.state == 'GO' and pink_area >= self.area_slow:
                self.trigger_cnt += 1; self.release_cnt = 0
                if self.trigger_cnt >= self.n_frames_trigger:
                    next_state = 'SLOW'
            elif self.state == 'SLOW':
                if pink_area >= self.area_stop:
                    self.trigger_cnt += 1; self.release_cnt = 0
                    if self.trigger_cnt >= self.n_frames_trigger:
                        next_state = 'STOP'
                elif pink_area <= self.area_resume:
                    self.release_cnt += 1; self.trigger_cnt = 0
                    if self.release_cnt >= self.n_frames_release:
                        next_state = 'GO'
                else:
                    self.trigger_cnt = 0; self.release_cnt = 0
            else:
                self.release_cnt += 1; self.trigger_cnt = 0
                if self.state != 'GO' and self.release_cnt >= self.n_frames_release:
                    next_state = 'GO'

            if next_state != self.state:
                self.get_logger().info(f'STATE {self.state} → {next_state}')
                self.state = next_state
                self.trigger_cnt = 0
                self.release_cnt = 0

            # --- 퍼블리시 정책 ---
            # GO: 아무것도 발행하지 않음(자동주행이 통과)
            # SLOW/STOP: /person_cmd_vel로 짧게 반복 발행(override 확실)
            if self.state == 'STOP':
                self.burst_publish(0.0)
            elif self.state == 'SLOW':
                self.burst_publish(self.v_slow)
            # else: GO → pass

            # 디버깅 뷰
            vis = frame.copy()
            cv2.rectangle(vis, (0, y0), (w-1, y1-1), (0, 255, 255), 2)
            cv2.imshow('Camera (ROI box)', vis)
            cv2.imshow('Pink Mask (ROI)', mask)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'이미지 콜백 오류: {e}')

    # 짧게 여러 번 내보내서 MUX override 보장
    def burst_publish(self, linear_x: float, repeats: int = 5, dt: float = 0.03):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = 0.0
        for _ in range(repeats):
            self.pub_person.publish(msg)
            time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

