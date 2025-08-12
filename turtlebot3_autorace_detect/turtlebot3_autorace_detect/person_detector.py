# person_detector.py (YOLO 적용 버전)
# 변경/추가 지점을 # [YOLO] 로 명시했습니다.

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        # ===== 설정값 =====
        # 입력 이미지 토픽 (autorace 파이프라인: 보정된 이미지)
        self.topic_image = '/camera/image_compensated'

        # [기존] 색상 기반 파라미터 (백업/비상용)
        self.lower_pink = np.array([140, 30, 80])
        self.upper_pink = np.array([175, 255, 255])

        # 상태머신 히스테리시스(프레임 누적) — 기존 로직 유지
        self.n_frames_trigger = 3   # 연속 N프레임 이상 조건 충족 시 상태 전이
        self.n_frames_release = 6   # 연속 N프레임 이상 조건 해제 시 복귀

        # ROI 방식
        self.roi_mode = 'custom'    # 'bottom' | 'center' | 'custom'
        self.roi_y_start_ratio = 0.35
        self.roi_y_end_ratio   = 0.75

        # [YOLO] 사용 스위치 및 파라미터
        self.use_yolo = True               # YOLO 사용 여부 (False면 색상 방식으로 복귀)
        self.yolo_conf = 0.4               # YOLO confidence threshold
        self.yolo_person_class_id = 0      # Ultralytics 모델에서 'person' 클래스 ID(일반적으로 0)
        self.yolo_box_area_stop_ratio = 0.02   # ROI 면적 대비 person box 합계 면적이 이 비율 이상이면 STOP
        self.yolo_box_area_slow_ratio = 0.01   # ROI 면적 대비 person box 합계 면적이 이 비율 이상이면 SLOW
        self.yolo_stop_on_any_person = True    # True면 박스 면적 상관없이 "사람 존재"만으로도 trigger 판단(ROI 내)

        # [색상 방식] 면적 임계값 (fallback/참고)
        self.area_slow    = 700
        self.area_stop    = 1000
        self.area_resume  = 500

        self.bridge = CvBridge()
        self.state = 'GO'  # 'GO' | 'SLOW' | 'STOP'
        self.trigger_cnt = 0
        self.release_cnt = 0

        # 구독/퍼블리시
        self.sub = self.create_subscription(Image, self.topic_image, self.image_cb, 10)

        # [REMIND] 기존에 직접 속도 퍼블리시는 off 권장. control_lane/safety_cmd_mux 체인 사용 시 Bool flag만 퍼블리시
        # self.pub_person = self.create_publisher(Twist, '/person_cmd_vel', 10)

        # 감지 플래그 (상태 기반): true = 감지/정지(or 감속) 필요
        self.pub_flag = self.create_publisher(Bool, '/person_detected', 10)

        # [YOLO] rqt_image_view용: 박스/라벨 그려진 결과 이미지 퍼블리시
        self.pub_debug_img = self.create_publisher(Image, '/person_detector/image_annotated', 10)

        # [YOLO] 모델 로딩
        self.yolo_model = None
        if self.use_yolo:
            try:
                # ultralytics 설치 필요: pip install ultralytics
                from ultralytics import YOLO
                # lightweight 모델 권장 (속도): 'yolov8n.pt' 또는 'yolov8s.pt'
                self.yolo_model = YOLO('yolov8n.pt')  # 모델 경로를 다른 곳으로 바꾸려면 여기 수정
                self.get_logger().info('[YOLO] YOLOv8 모델 로딩 성공: yolov8n.pt')
            except Exception as e:
                self.get_logger().error(f'[YOLO] 모델 로딩 실패. 색상 방식으로 fallback 합니다: {e}')
                self.use_yolo = False
                self.yolo_model = None

        self.get_logger().info('PersonDetector ready.')

    def get_roi(self, frame):
        """ROI 영역을 리턴: (roi_img, (y0, y1, w, h))"""
        h, w = frame.shape[:2]
        if self.roi_mode == 'bottom':
            y0 = int(h * 0.55)
            y1 = int(h * 1.00)
        elif self.roi_mode == 'center':
            y0 = int(h * 0.30)
            y1 = int(h * 0.70)
        else:
            r0 = max(0.0, min(self.roi_y_start_ratio, 1.0))
            r1 = max(r0 + 0.05, min(self.roi_y_end_ratio, 1.0))
            y0 = int(h * r0)
            y1 = int(h * r1)
        roi = frame[y0:y1, :]
        return roi, (y0, y1, w, h)

    def publish_flag(self):
        msg = Bool()
        msg.data = (self.state != 'GO')  # STOP 또는 SLOW 이면 True
        self.pub_flag.publish(msg)

    def _draw_roi(self, img, y0, y1, color=(0, 255, 255)):
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, y0), (w - 1, y1 - 1), color, 2)

    def _annotate_and_publish(self, frame_bgr, dets_in_roi):
        """rqt에서 확인할 수 있도록 주석 이미지를 퍼블리시"""
        vis = frame_bgr.copy()
        # ROI 박스
        roi_img, (y0, y1, w, h) = self.get_roi(frame_bgr)
        self._draw_roi(vis, y0, y1)

        # 감지 박스 그리기
        for (x1, y1_box, x2, y2_box, conf) in dets_in_roi:
            cv2.rectangle(vis, (x1, y1_box), (x2, y2_box), (0, 255, 0), 2)
            label = f'person {conf:.2f}'
            cv2.putText(vis, label, (x1, max(y1_box - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 현재 상태 오버레이
        cv2.putText(vis, f'STATE: {self.state}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if self.state == 'STOP' else ((0, 165, 255) if self.state == 'SLOW' else (255, 255, 255)),
                    2)

        # 퍼블리시
        try:
            self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            self.get_logger().warn(f'annotated image publish error: {e}')

    # [YOLO] ROI 내 person 감지 → (박스리스트, ROI 대비 박스합 면적비, 존재여부)
    def _yolo_person_in_roi(self, frame_bgr):
        if self.yolo_model is None:
            return [], 0.0, False

        roi_img, (y0, y1, w, h) = self.get_roi(frame_bgr)
        roi_area = max(1, roi_img.shape[0] * roi_img.shape[1])

        # 모델 추론
        try:
            # 결과는 xyxy, conf, cls 등 포함
            results = self.yolo_model(roi_img, conf=self.yolo_conf, verbose=False)
        except Exception as e:
            self.get_logger().error(f'[YOLO] 추론 실패: {e}')
            return [], 0.0, False

        boxes = []
        total_person_area = 0
        person_found = False

        # Ultralytics 결과 파싱
        for r in results:
            if not hasattr(r, 'boxes'):
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                if cls_id != self.yolo_person_class_id:
                    continue
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].cpu().numpy().astype(int)  # [x1,y1,x2,y2] in ROI coords
                x1, y1_box, x2, y2_box = xyxy.tolist()
                # ROI 좌표를 원본 좌표로 보정해서 저장 (주석 그리기 용)
                boxes.append((x1, y1_box + y0, x2, y2_box + y0, conf))
                person_found = True
                # ROI 내 면적 합산 (비율 판단용)
                w_box = max(0, x2 - x1)
                h_box = max(0, y2_box - y1_box)
                total_person_area += (w_box * h_box)

        area_ratio = total_person_area / float(roi_area)
        return boxes, area_ratio, person_found

    def image_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # ROI 추출
            roi, (y0, y1, w, h) = self.get_roi(frame)

            # ====== 핵심 판단 ======
            if self.use_yolo:
                # [YOLO] person 박스 탐지
                det_boxes, area_ratio, person_found = self._yolo_person_in_roi(frame)

                # 상태머신 입력 신호 만들기
                input_slow = (area_ratio >= self.yolo_box_area_slow_ratio)
                input_stop = (area_ratio >= self.yolo_box_area_stop_ratio)
                if self.yolo_stop_on_any_person and person_found:
                    # ROI 안에 사람만 있으면 바로 트리거 카운트 증가
                    input_slow = True
                    # STOP 여부는 면적 비율 기준으로만 결정(급정지 방지)
                # 디버그
                self.get_logger().info(f'[YOLO] area_ratio={area_ratio:.4f}, person_found={person_found}, state={self.state}')

                # === 상태머신 === (기존 로직 최대한 유지)
                next_state = self.state
                if self.state != 'STOP' and input_stop:
                    self.trigger_cnt += 1; self.release_cnt = 0
                    if self.trigger_cnt >= self.n_frames_trigger:
                        next_state = 'STOP'
                elif self.state == 'STOP' and not (input_slow or input_stop):
                    self.release_cnt += 1; self.trigger_cnt = 0
                    if self.release_cnt >= self.n_frames_release:
                        next_state = 'GO'
                elif self.state == 'GO' and input_slow:
                    self.trigger_cnt += 1; self.release_cnt = 0
                    if self.trigger_cnt >= self.n_frames_trigger:
                        next_state = 'SLOW'
                elif self.state == 'SLOW':
                    if input_stop:
                        self.trigger_cnt += 1; self.release_cnt = 0
                        if self.trigger_cnt >= self.n_frames_trigger:
                            next_state = 'STOP'
                    elif not input_slow:  # 완전히 해제되어야 GO
                        self.release_cnt += 1; self.trigger_cnt = 0
                        if self.release_cnt >= self.n_frames_release:
                            next_state = 'GO'
                    else:
                        self.trigger_cnt = 0; self.release_cnt = 0
                else:
                    # 기타 안정화
                    self.release_cnt += 1; self.trigger_cnt = 0
                    if self.state != 'GO' and self.release_cnt >= self.n_frames_release:
                        next_state = 'GO'

                # 상태 변경 반영
                if next_state != self.state:
                    self.get_logger().info(f'STATE {self.state} → {next_state}')
                    self.state = next_state
                    self.trigger_cnt = 0
                    self.release_cnt = 0

                # 플래그 퍼블리시 (매 프레임)
                self.publish_flag()

                # rqt 확인용 이미지 주석 퍼블리시
                self._annotate_and_publish(frame, det_boxes)

            else:
                # ===== [색상 방식] Fallback =====
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, self.lower_pink, self.upper_pink)
                pink_area = int(cv2.countNonZero(mask))
                self.get_logger().info(f'[COLOR] 핑크 영역(ROI): {pink_area} | state={self.state}')

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

                self.publish_flag()

                # rqt 확인용(색상 방식) — ROI 박스만 그려서 퍼블리시
                vis = frame.copy()
                self._draw_roi(vis, y0, y1, color=(0, 255, 255))
                cv2.putText(vis, f'STATE: {self.state}', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255) if self.state == 'STOP' else ((0, 165, 255) if self.state == 'SLOW' else (255, 255, 255)),
                            2)
                self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))

        except Exception as e:
            self.get_logger().error(f'이미지 콜백 오류: {e}')

    # (선택) 기존처럼 Twist를 짧게 여러 번 퍼블리시하고 싶다면 사용
    def burst_publish(self, linear_x: float, repeats: int = 5, dt: float = 0.03):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = 0.0
        # if hasattr(self, 'pub_person'):
        #     for _ in range(repeats):
        #         self.pub_person.publish(msg)
        #         time.sleep(dt)

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

