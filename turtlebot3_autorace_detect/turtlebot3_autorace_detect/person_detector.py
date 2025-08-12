# person_detector.py (YOLO only)
# 컬러기반(Fallback) 제거, YOLO만 사용
# GPU 이슈 회피용 기본 device=CPU 지정 (# [DEVICE])

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

        # ===== 입력 이미지 토픽 =====
        self.topic_image = '/camera/image_compensated'

        # 상태머신 히스테리시스(프레임 누적)
        self.n_frames_trigger = 2   # 연속 N프레임 충족 시 상태 전이
        self.n_frames_release = 6   # 연속 N프레임 해제 시 복귀

        # ROI (로봇 앞쪽만 보기)
        self.roi_mode = 'custom'    # 'bottom' | 'center' | 'custom'
        self.roi_y_start_ratio = 0.35
        self.roi_y_end_ratio   = 0.75

        # [YOLO] 파라미터
        self.use_yolo = True
        self.yolo_conf = 0.7
        self.yolo_person_class_id = 0
        self.yolo_box_area_stop_ratio = 0.04   # ROI 면적 대비 person 박스 합계가 이 비율↑면 STOP
        self.yolo_box_area_slow_ratio = 0.02   # ROI 면적 대비 person 박스 합계가 이 비율↑면 SLOW
        self.yolo_stop_on_any_person = False    # ROI 내 사람이 보이면(면적 무관) SLOW 트리거

        self.slow_release_ratio = 0.008         # 0.8% 미만 6프레임이면 GO로 복귀
        self.stop_release_ratio = 0.02          # (선택) STOP 해제 문턱
       
        # [DEVICE] 기본 CPU 강제 (MX250 CUDA 커널 불일치 방지)
        self.yolo_device = 'cpu'  # 필요 시 'cuda:0' 로 변경

        self.bridge = CvBridge()
        self.state = 'GO'  # 'GO' | 'SLOW' | 'STOP'
        self.trigger_cnt = 0
        self.release_cnt = 0

        # 구독/퍼블리시
        self.sub = self.create_subscription(Image, self.topic_image, self.image_cb, 1)
        self.pub_flag = self.create_publisher(Bool, '/person_detected', 1)
        self.pub_debug_img = self.create_publisher(Image, '/person_detector/image_annotated', 1)

        # [YOLO] 모델 로딩
        self.yolo_model = None
        if self.use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO('yolov8n.pt')
                # 디바이스 지정
                try:
                    self.yolo_model.to(self.yolo_device)
                except Exception as dev_e:
                    self.get_logger().warn(f'[YOLO] device 이동 실패({dev_e}), CPU로 전환')
                    self.yolo_device = 'cpu'
                    self.yolo_model.to('cpu')
                self.get_logger().info(f'[YOLO] 모델 로딩 성공 (device={self.yolo_device})')
            except Exception as e:
                self.get_logger().error(f'[YOLO] 모델 로딩 실패: {e}')
                self.use_yolo = False
                self.yolo_model = None

        self.get_logger().info('PersonDetector ready (YOLO only).')

    def get_roi(self, frame):
        """ROI 영역을 리턴: (roi_img, (y0, y1, w, h))"""
        h, w = frame.shape[:2]
        if self.roi_mode == 'bottom':
            y0 = int(h * 0.55); y1 = int(h * 1.00)
        elif self.roi_mode == 'center':
            y0 = int(h * 0.30); y1 = int(h * 0.70)
        else:
            r0 = max(0.0, min(self.roi_y_start_ratio, 1.0))
            r1 = max(r0 + 0.05, min(self.roi_y_end_ratio, 1.0))
            y0 = int(h * r0); y1 = int(h * r1)
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
        roi_img, (y0, y1, w, h) = self.get_roi(frame_bgr)
        self._draw_roi(vis, y0, y1)

        for (x1, y1_box, x2, y2_box, conf) in dets_in_roi:
            cv2.rectangle(vis, (x1, y1_box), (x2, y2_box), (0, 255, 0), 2)
            label = f'person {conf:.2f}'
            cv2.putText(vis, label, (x1, max(y1_box - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(vis, f'STATE: {self.state}', (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255) if self.state == 'STOP'
                    else ((0, 165, 255) if self.state == 'SLOW' else (255, 255, 255)), 2)

        try:
            self.pub_debug_img.publish(self.bridge.cv2_to_imgmsg(vis, encoding='bgr8'))
        except Exception as e:
            self.get_logger().warn(f'annotated image publish error: {e}')

    # [YOLO] ROI 내 person 감지 → (박스 리스트, ROI 대비 박스합 면적비, 존재여부)
    def _yolo_person_in_roi(self, frame_bgr):
        if self.yolo_model is None:
            return [], 0.0, False

        roi_img, (y0, y1, w, h) = self.get_roi(frame_bgr)
        roi_area = max(1, roi_img.shape[0] * roi_img.shape[1])

        try:
            results = self.yolo_model(
                roi_img, conf=self.yolo_conf, verbose=False, device=self.yolo_device
            )
        except Exception as e:
            self.get_logger().error(f'[YOLO] 추론 실패(device={self.yolo_device}): {e}')
            # GPU 실패 시 CPU 폴백
            if self.yolo_device != 'cpu':
                try:
                    self.get_logger().warn('[YOLO] CPU로 자동 전환합니다.')
                    self.yolo_device = 'cpu'
                    self.yolo_model.to('cpu')
                    results = self.yolo_model(roi_img, conf=self.yolo_conf, verbose=False, device='cpu')
                except Exception as e2:
                    self.get_logger().error(f'[YOLO] CPU 추론도 실패: {e2}')
                    return [], 0.0, False
            else:
                return [], 0.0, False

        boxes, total_person_area, person_found = [], 0, False

        for r in results:
            if not hasattr(r, 'boxes'):
                continue
            for b in r.boxes:
                cls_id = int(b.cls.item()) if b.cls is not None else -1
                if cls_id != self.yolo_person_class_id:
                    continue
                conf = float(b.conf.item()) if b.conf is not None else 0.0
                xyxy = b.xyxy[0].cpu().numpy().astype(int)  # [x1,y1,x2,y2] (ROI 좌표계)
                x1, y1_box, x2, y2_box = xyxy.tolist()

                # 원본 좌표계로 보정(시각화용)
                boxes.append((x1, y1_box + y0, x2, y2_box + y0, conf))
                person_found = True

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

            # ====== YOLO 전용 판단 ======
            if not self.use_yolo:
                self.get_logger().error('YOLO가 비활성화되어 있습니다. (색상 방식은 제거됨)')
                return

            det_boxes, area_ratio, person_found = self._yolo_person_in_roi(frame)

            # 상태머신 입력 신호
            input_slow = (area_ratio >= self.yolo_box_area_slow_ratio)
            input_stop = (area_ratio >= self.yolo_box_area_stop_ratio)
            if self.yolo_stop_on_any_person and person_found:
                input_slow = True  # 존재만으로 감속(급정지 방지용)
            self.get_logger().info(f'[YOLO] area_ratio={area_ratio:.4f}, person_found={person_found}, state={self.state}')

            # === 상태머신 ===
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
                elif not input_slow:
                    self.release_cnt += 1; self.trigger_cnt = 0
                    if self.release_cnt >= self.n_frames_release:
                        next_state = 'GO'
                else:
                    self.trigger_cnt = 0; self.release_cnt = 0
            # else:
            #     self.release_cnt += 1; self.trigger_cnt = 0
            #     if self.state != 'GO' and self.release_cnt >= self.n_frames_release:
            #         next_state = 'GO'

            # 상태 변경 반영
            if next_state != self.state:
                self.get_logger().info(f'STATE {self.state} → {next_state}')
                self.state = next_state
                self.trigger_cnt = 0
                self.release_cnt = 0

            # 플래그 퍼블리시
            self.publish_flag()

            # rqt 확인용 이미지 퍼블리시
            self._annotate_and_publish(frame, det_boxes)

        except Exception as e:
            self.get_logger().error(f'이미지 콜백 오류: {e}')

    # (선택) 필요 시 사용
    def burst_publish(self, linear_x: float, repeats: int = 5, dt: float = 0.03):
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = 0.0
        # 퍼블리셔를 쓰는 구조가 아니므로 주석 유지

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


