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
# Authors:
#   - Leon Jung, Gilbert, Ashe Kim, Hyungyu Kim, ChanHyeong Lee
#   - Special Thanks : Roger Sacchelli

import cv2
from cv_bridge import CvBridge
import numpy as np
from rcl_interfaces.msg import IntegerRange
from rcl_interfaces.msg import ParameterDescriptor
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Bool
from std_msgs.msg import Bool, UInt8


BASE_FRACTION = 3000

class DetectLane(Node):

    def __init__(self):
        super().__init__('detect_lane')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_jpeg/compressed',
            self.cbFindLane,
            10)
        self.subscription

        self.pub_stop_line = self.create_publisher(Bool, '/detect/stop_line', 10)

        parameter_descriptor_hue = ParameterDescriptor(
            description='hue parameter range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=179,
                step=1)]
        )
        parameter_descriptor_saturation_lightness = ParameterDescriptor(
            description='saturation and lightness range',
            integer_range=[IntegerRange(
                from_value=0,
                to_value=255,
                step=1)]
        )
        self.declare_parameters(
            namespace='',
            parameters=[
                ('detect.lane.white.hue_l', 0,
                    parameter_descriptor_hue),
                ('detect.lane.white.hue_h', 179,
                    parameter_descriptor_hue),
                ('detect.lane.white.saturation_l', 0,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.saturation_h', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_l', 105,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.white.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.hue_l', 10,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.hue_h', 127,
                    parameter_descriptor_hue),
                ('detect.lane.yellow.saturation_l', 70,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.saturation_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_l', 95,
                    parameter_descriptor_saturation_lightness),
                ('detect.lane.yellow.lightness_h', 255,
                    parameter_descriptor_saturation_lightness),
                ('is_detection_calibration_mode', False)
            ]
        )

        self.hue_white_l = self.get_parameter(
            'detect.lane.white.hue_l').get_parameter_value().integer_value
        self.hue_white_h = self.get_parameter(
            'detect.lane.white.hue_h').get_parameter_value().integer_value
        self.saturation_white_l = self.get_parameter(
            'detect.lane.white.saturation_l').get_parameter_value().integer_value
        self.saturation_white_h = self.get_parameter(
            'detect.lane.white.saturation_h').get_parameter_value().integer_value
        self.lightness_white_l = self.get_parameter(
            'detect.lane.white.lightness_l').get_parameter_value().integer_value
        self.lightness_white_h = self.get_parameter(
            'detect.lane.white.lightness_h').get_parameter_value().integer_value

        self.hue_yellow_l = self.get_parameter(
            'detect.lane.yellow.hue_l').get_parameter_value().integer_value
        self.hue_yellow_h = self.get_parameter(
            'detect.lane.yellow.hue_h').get_parameter_value().integer_value
        self.saturation_yellow_l = self.get_parameter(
            'detect.lane.yellow.saturation_l').get_parameter_value().integer_value
        self.saturation_yellow_h = self.get_parameter(
            'detect.lane.yellow.saturation_h').get_parameter_value().integer_value
        self.lightness_yellow_l = self.get_parameter(
            'detect.lane.yellow.lightness_l').get_parameter_value().integer_value
        self.lightness_yellow_h = self.get_parameter(
            'detect.lane.yellow.lightness_h').get_parameter_value().integer_value

        self.is_calibration_mode = self.get_parameter(
            'is_detection_calibration_mode').get_parameter_value().bool_value
        if self.is_calibration_mode:
            self.add_on_set_parameters_callback(self.cbGetDetectLaneParam)

        self.sub_image_type = 'raw'         # you can choose image type 'compressed', 'raw'
        self.pub_image_type = 'compressed'  # you can choose image type 'compressed', 'raw'

        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.cbFindLane, 1
                )
        elif self.sub_image_type == 'raw':
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.cbFindLane, 1
                )

        if self.pub_image_type == 'compressed':
            self.pub_image_lane = self.create_publisher(
                CompressedImage, '/detect/image_output/compressed', 1
                )
        elif self.pub_image_type == 'raw':
            self.pub_image_lane = self.create_publisher(
                Image, '/detect/image_output', 1
                )
        self.pub_dashed = self.create_publisher(Bool, '/detect/dashed_line', 1)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub1/compressed', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    CompressedImage, '/detect/image_output_sub2/compressed', 1
                    )
            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane = self.create_publisher(
                    Image, '/detect/image_output_sub1', 1
                    )
                self.pub_image_yellow_lane = self.create_publisher(
                    Image, '/detect/image_output_sub2', 1
                    )

        self.pub_lane = self.create_publisher(Float64, '/detect/lane', 1)


        self.pub_yellow_line_reliability = self.create_publisher(
            UInt8, '/detect/yellow_line_reliability', 1
            )

        self.pub_white_line_reliability = self.create_publisher(
            UInt8, '/detect/white_line_reliability', 1
            )

        self.pub_lane_state = self.create_publisher(UInt8, '/detect/lane_state', 1)

        self.cvBridge = CvBridge()

        self.counter = 1

        self.window_width = 1000.
        self.window_height = 600.

        self.reliability_white_line = 100
        self.reliability_yellow_line = 100

        self.mov_avg_left = np.empty((0, 3))
        self.mov_avg_right = np.empty((0, 3))

        self.dashed_counter = {"LEFT": 0, "RIGHT": 0}
        self.dashed_window = 5  # 최대 카운트 누적 윈도우
        self.dashed_threshold = 1  # 임계값: 이 값 이상이면 점선으로 판단

        self.detect_dot_flag = False

        self.pre_centerx = 500

    def detect_stop_line(self, image):
        # 개선된 ROI: y:420~500 (위로 약간 올림), x:200~800
        roi = image[420:500, 200:800]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 개선된 threshold (더 어두운 흰색도 포함)
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # 개선된 Morphology (더 작은 커널)
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 개선된 HoughLinesP 파라미터
        lines = cv2.HoughLinesP(
            processed,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=30,
            maxLineGap=5
        )

        count_horizontal = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1)) * 180.0 / np.pi
                if angle < 15:  # 수평선 각도 기준도 살짝 완화
                    count_horizontal += 1

        return count_horizontal >= 2  # 감지 기준도 완화


    def cbGetDetectLaneParam(self, parameters):
        for param in parameters:
            self.get_logger().info(f'Parameter name: {param.name}')
            self.get_logger().info(f'Parameter value: {param.value}')
            self.get_logger().info(f'Parameter type: {param.type_}')
            if param.name == 'detect.lane.white.hue_l':
                self.hue_white_l = param.value
            elif param.name == 'detect.lane.white.hue_h':
                self.hue_white_h = param.value
            elif param.name == 'detect.lane.white.saturation_l':
                self.saturation_white_l = param.value
            elif param.name == 'detect.lane.white.saturation_h':
                self.saturation_white_h = param.value
            elif param.name == 'detect.lane.white.lightness_l':
                self.lightness_white_l = param.value
            elif param.name == 'detect.lane.white.lightness_h':
                self.lightness_white_h = param.value
            elif param.name == 'detect.lane.yellow.hue_l':
                self.hue_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.hue_h':
                self.hue_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.saturation_l':
                self.saturation_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.saturation_h':
                self.saturation_yellow_h = param.value
            elif param.name == 'detect.lane.yellow.lightness_l':
                self.lightness_yellow_l = param.value
            elif param.name == 'detect.lane.yellow.lightness_h':
                self.lightness_yellow_h = param.value
            return SetParametersResult(successful=True)

    def cbFindLane(self, image_msg):
        # Change the frame rate by yourself. Now, it is set to 1/3 (10fps).
        # Unappropriate value of frame rate may cause huge delay on entire recognition process.
        # This is up to your computer's operating power.
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        elif self.sub_image_type == 'raw':
            cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')

        white_fraction, cv_white_lane = self.maskWhiteLane(cv_image)
        yellow_fraction, cv_yellow_lane = self.maskYellowLane(cv_image)

        try:
            if not self.detect_dot_flag:
                is_dashed = self.is_dashed_line(cv_white_lane, min_len=2, max_len=100, min_segments=2, std_threshold=25, visualize=False)
                self.get_logger().warn(f"점선 인식 결과: {is_dashed}")
                if is_dashed:
                    self.detect_dot_flag = True
                    self.prefer_left_lane = True  # ✅ 점선 감지 시 왼쪽 차선 기준 사용
                    self.yellow_line_miss_count = 0
                msg = Bool()
                msg.data = is_dashed
                self.pub_dashed.publish(msg)

        except Exception as e:
            self.get_logger().error(f"점선 인식 중 오류: {e}")
        try:
            if yellow_fraction > BASE_FRACTION:
                self.left_fitx, self.left_fit = self.fit_from_lines(
                    self.left_fit, cv_yellow_lane)
                self.mov_avg_left = np.append(
                    self.mov_avg_left, np.array([self.left_fit]), axis=0
                    )

            if white_fraction > BASE_FRACTION:
                self.right_fitx, self.right_fit = self.fit_from_lines(
                    self.right_fit, cv_white_lane)
                self.mov_avg_right = np.append(
                    self.mov_avg_right, np.array([self.right_fit]), axis=0
                    )
        except Exception:
            if yellow_fraction > BASE_FRACTION:
                self.left_fitx, self.left_fit = self.sliding_windown(cv_yellow_lane, 'left')
                self.mov_avg_left = np.array([self.left_fit])

            if white_fraction > BASE_FRACTION:
                self.right_fitx, self.right_fit = self.sliding_windown(cv_white_lane, 'right')
                self.mov_avg_right = np.array([self.right_fit])

        MOV_AVG_LENGTH = 5

        self.left_fit = np.array([
            np.mean(self.mov_avg_left[::-1][:, 0][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_left[::-1][:, 1][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_left[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])
        self.right_fit = np.array([
            np.mean(self.mov_avg_right[::-1][:, 0][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_right[::-1][:, 1][0:MOV_AVG_LENGTH]),
            np.mean(self.mov_avg_right[::-1][:, 2][0:MOV_AVG_LENGTH])
            ])

        if self.mov_avg_left.shape[0] > 1000:
            self.mov_avg_left = self.mov_avg_left[0:MOV_AVG_LENGTH]

        if self.mov_avg_right.shape[0] > 1000:
            self.mov_avg_right = self.mov_avg_right[0:MOV_AVG_LENGTH]

        self.make_lane(cv_image, white_fraction, yellow_fraction)

        is_stopline = self.detect_stop_line(cv_image)
        stopline_msg = Bool()
        stopline_msg.data = is_stopline
        self.pub_stop_line.publish(stopline_msg)

        if is_stopline:
            self.get_logger().info("Stop Line Detected)")

    def maskWhiteLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_white_l
        Hue_h = self.hue_white_h
        Saturation_l = self.saturation_white_l
        Saturation_h = self.saturation_white_h
        Lightness_l = self.lightness_white_l
        Lightness_h = self.lightness_white_h

        lower_white = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_white = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_white, upper_white)

        fraction_num = np.count_nonzero(mask)

        if not self.is_calibration_mode:
            if fraction_num > 35000:
                if self.lightness_white_l < 250:
                    self.lightness_white_l += 5
            elif fraction_num < 5000:
                if self.lightness_white_l > 50:
                    self.lightness_white_l -= 5

        how_much_short = 0

        for i in range(0, 600):
            if np.count_nonzero(mask[i, ::]) > 0:
                how_much_short += 1

        how_much_short = 600 - how_much_short

        if how_much_short > 100:
            if self.reliability_white_line >= 5:
                self.reliability_white_line -= 5
        elif how_much_short <= 100:
            if self.reliability_white_line <= 99:
                self.reliability_white_line += 5

        msg_white_line_reliability = UInt8()
        msg_white_line_reliability.data = self.reliability_white_line
        self.pub_white_line_reliability.publish(msg_white_line_reliability)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_white_lane.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
                    )

            elif self.pub_image_type == 'raw':
                self.pub_image_white_lane.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
                    )

        return fraction_num, mask

    def maskYellowLane(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        Hue_l = self.hue_yellow_l
        Hue_h = self.hue_yellow_h
        Saturation_l = self.saturation_yellow_l
        Saturation_h = self.saturation_yellow_h
        Lightness_l = self.lightness_yellow_l
        Lightness_h = self.lightness_yellow_h

        lower_yellow = np.array([Hue_l, Saturation_l, Lightness_l])
        upper_yellow = np.array([Hue_h, Saturation_h, Lightness_h])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        fraction_num = np.count_nonzero(mask)

        if not self.is_calibration_mode:
            if fraction_num > 35000:
                if self.lightness_yellow_l < 250:
                    self.lightness_yellow_l += 20
            elif fraction_num < 5000:
                if self.lightness_yellow_l > 90:
                    self.lightness_yellow_l -= 20

        how_much_short = 0

        for i in range(0, 600):
            if np.count_nonzero(mask[i, ::]) > 0:
                how_much_short += 1

        how_much_short = 600 - how_much_short

        if how_much_short > 100:
            if self.reliability_yellow_line >= 5:
                self.reliability_yellow_line -= 5
        elif how_much_short <= 100:
            if self.reliability_yellow_line <= 99:
                self.reliability_yellow_line += 5

        msg_yellow_line_reliability = UInt8()
        msg_yellow_line_reliability.data = self.reliability_yellow_line
        self.pub_yellow_line_reliability.publish(msg_yellow_line_reliability)

        if self.is_calibration_mode:
            if self.pub_image_type == 'compressed':
                self.pub_image_yellow_lane.publish(
                    self.cvBridge.cv2_to_compressed_imgmsg(mask, 'jpg')
                    )

            elif self.pub_image_type == 'raw':
                self.pub_image_yellow_lane.publish(
                    self.cvBridge.cv2_to_imgmsg(mask, 'bgr8')
                    )

        return fraction_num, mask

    def fit_from_lines(self, lane_fit, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        lane_inds = (
            (nonzerox >
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] - margin)) &
            (nonzerox <
                (lane_fit[0] * (nonzeroy ** 2) + lane_fit[1] * nonzeroy + lane_fit[2] + margin))
                )

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        lane_fit = np.polyfit(y, x, 2)

        ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit

    def sliding_windown(self, img_w, left_or_right):
        histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)

        out_img = np.dstack((img_w, img_w, img_w)) * 255

        midpoint = np.int_(histogram.shape[0] / 2)

        if left_or_right == 'left':
            lane_base = np.argmax(histogram[:midpoint])
        elif left_or_right == 'right':
            lane_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 20

        window_height = np.int_(img_w.shape[0] / nwindows)

        nonzero = img_w.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        x_current = lane_base

        margin = 50

        minpix = 50

        lane_inds = []

        for window in range(nwindows):
            win_y_low = img_w.shape[0] - (window + 1) * window_height
            win_y_high = img_w.shape[0] - window * window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            cv2.rectangle(
                out_img, (win_x_low, win_y_low), (win_x_high, win_y_high), (0, 255, 0), 2)

            good_lane_inds = (
                (nonzeroy >= win_y_low) &
                (nonzeroy < win_y_high) &
                (nonzerox >= win_x_low) &
                (nonzerox < win_x_high)
                ).nonzero()[0]

            lane_inds.append(good_lane_inds)

            if len(good_lane_inds) > minpix:
                x_current = np.int_(np.mean(nonzerox[good_lane_inds]))

        lane_inds = np.concatenate(lane_inds)

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        try:
            lane_fit = np.polyfit(y, x, 2)
            self.lane_fit_bef = lane_fit
        except Exception:
            lane_fit = self.lane_fit_bef

        ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
        lane_fitx = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]

        return lane_fitx, lane_fit


    def is_dashed_line(self, mask, min_len=5, max_len=80, min_segments=3, std_threshold=15, visualize=False):
        """
        좌/우 차선 영역을 확인하여 점선(2차선)이 존재하는지 판단합니다.

        Returns:
            bool: 좌/우 중 한 쪽이라도 점선이면 True
        """
        if mask is None or not isinstance(mask, np.ndarray) or len(mask.shape) != 2:
            self.get_logger().error("Invalid mask input for dashed line detection.")
            return False

        height, width = mask.shape

        # 좌/우 ROI 설정
        roi_left = mask[int(height * 0.4):int(height * 0.95), int(width * 0.1):int(width * 0.45)]
        roi_right = mask[int(height * 0.4):int(height * 0.95), int(width * 0.55):int(width * 0.9)]

        left_result = self._check_dashed_roi(roi_left, min_len, max_len, min_segments, std_threshold, visualize, "LEFT")
        right_result = self._check_dashed_roi(roi_right, min_len, max_len, min_segments, std_threshold, visualize, "RIGHT")

        result = left_result or right_result
        self.get_logger().warn(f"점선 인식 결과: {result}")
        return result


    def _check_dashed_roi(self, roi, min_len, max_len, min_segments, std_threshold, visualize, label="ROI"):
        """
        특정 ROI에서 점선을 판단하고, 일정 횟수 이상일 때만 True 반환.
        """
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dashed_bounding_boxes = []
        y_positions = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            length = max(w, h)
            if min_len < length < max_len:
                dashed_bounding_boxes.append((x, y, w, h))
                y_positions.append(y)

        if visualize:
            debug_img = cv2.cvtColor(roi.copy(), cv2.COLOR_GRAY2BGR)
            for x, y, w, h in dashed_bounding_boxes:
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow(f"Dashed Contours Debug - {label}", debug_img)
            cv2.waitKey(1)

        # === 판단 기준 ===
        is_dashed_now = False

        if len(y_positions) >= min_segments:
            y_positions.sort()
            y_gaps = np.diff(y_positions)
            std_gap = np.std(y_gaps) if len(y_gaps) > 0 else 0
            self.get_logger().info(f"[{label}] 점선 조각: {len(y_positions)}, 간격 std: {std_gap:.2f}")

            if std_gap < std_threshold:
                is_dashed_now = True
        else:
            self.get_logger().warn(f"[{label}] 점선 조각 부족: {len(y_positions)}개")

        # === 카운터 누적 ===
        if is_dashed_now:
            self.dashed_counter[label] += 1
        else:
            self.dashed_counter[label] = max(0, self.dashed_counter[label] - 1)

        self.dashed_counter[label] = min(self.dashed_counter[label], self.dashed_window)

        if self.dashed_counter[label] >= self.dashed_threshold:
            self.get_logger().info(f"[{label}] 누적 카운트 충족: {self.dashed_counter[label]}")
            return True
        else:
            return False


    def make_lane(self, cv_image, white_fraction, yellow_fraction):
        # Create an image to draw the lines on
        warp_zero = np.zeros((cv_image.shape[0], cv_image.shape[1], 1), dtype=np.uint8)

        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        color_warp_lines = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, cv_image.shape[0] - 1, cv_image.shape[0])

        # both lane -> 2, left lane -> 1, right lane -> 3, none -> 0
        lane_state = UInt8()

        self.get_logger().info(f"yellow: {yellow_fraction}, white: {white_fraction}")
        self.get_logger().info(
            f"yellow_rel={self.reliability_yellow_line}, white_rel={self.reliability_white_line}"
        )

        if yellow_fraction > BASE_FRACTION:
            pts_left = np.array([np.flipud(np.transpose(np.vstack([self.left_fitx, ploty])))])
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_left]),
                isClosed=False,
                color=(0, 0, 255),
                thickness=25
                )

        if white_fraction > BASE_FRACTION:
            pts_right = np.array([np.transpose(np.vstack([self.right_fitx, ploty]))])
            cv2.polylines(
                color_warp_lines,
                np.int_([pts_right]),
                isClosed=False,
                color=(255, 255, 0),
                thickness=25
                )

        self.is_center_x_exist = True

        LANE_WIDTH = 280

        if self.reliability_white_line > 50 and self.reliability_yellow_line > 50:
            if white_fraction > BASE_FRACTION and yellow_fraction > BASE_FRACTION:
                centerx = np.mean([self.left_fitx, self.right_fitx], axis=0)
                pts = np.hstack((pts_left, pts_right))
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 2

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

                # Draw the lane onto the warped blank image
                cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

            if white_fraction > BASE_FRACTION and yellow_fraction <= BASE_FRACTION:
                centerx = np.subtract(self.right_fitx, 280)
                # if getattr(self, "prefer_left_lane", False):  # ← 왼쪽 변경 중이면
                #     centerx = np.subtract(self.right_fitx, LANE_WIDTH * 2 // 2)  # 왼쪽 차선 추정 쪽으로 더 왼쪽
                # else:
                #     centerx = np.subtract(self.right_fitx, LANE_WIDTH)           # 기존 로직(오른쪽 차선 중심)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 3

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

            if white_fraction <= BASE_FRACTION and yellow_fraction > BASE_FRACTION:
                centerx = np.add(self.left_fitx, 280)
                # if getattr(self, "prefer_left_lane", False):  # ← 왼쪽 변경 중이면
                #     centerx = np.subtract(self.left_fitx, LANE_WIDTH)            # 노란선의 왼쪽(=왼쪽 차선 중심)
                # else:
                #     centerx = np.add(self.left_fitx, LANE_WIDTH)                 # 기존 로직(오른쪽 차선 중심)
                pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

                lane_state.data = 1

                cv2.polylines(
                    color_warp_lines,
                    np.int_([pts_center]),
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=12
                    )

        elif self.reliability_white_line <= 50 and self.reliability_yellow_line > 50:
            centerx = np.add(self.left_fitx, 280)
            # if getattr(self, "prefer_left_lane", False):  # ← 왼쪽 변경 중이면
            #     centerx = np.subtract(self.left_fitx, LANE_WIDTH)            # 노란선의 왼쪽(=왼쪽 차선 중심)
            # else:
            #     centerx = np.add(self.left_fitx, LANE_WIDTH)                 # 기존 로직(오른쪽 차선 중심)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 1

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        elif self.reliability_white_line > 50 and self.reliability_yellow_line <= 50:
            centerx = np.subtract(self.right_fitx, 280)
            # if getattr(self, "prefer_left_lane", False):  # ← 왼쪽 변경 중이면
            #     centerx = np.subtract(self.right_fitx, LANE_WIDTH * 2 // 2)  # 왼쪽 차선 추정 쪽으로 더 왼쪽
            # else:
            #     centerx = np.subtract(self.right_fitx, LANE_WIDTH)           # 기존 로직(오른쪽 차선 중심)
            pts_center = np.array([np.transpose(np.vstack([centerx, ploty]))])

            lane_state.data = 3

            cv2.polylines(
                color_warp_lines,
                np.int_([pts_center]),
                isClosed=False,
                color=(0, 255, 255),
                thickness=12
                )

        else:
            self.is_center_x_exist = False
            # self.is_center_x_exist = True

            lane_state.data = 0

            pass

        self.pub_lane_state.publish(lane_state)
        self.get_logger().info(f'Lane state: {lane_state.data}')

        # Combine the result with the original image
        final = cv2.addWeighted(cv_image, 1, color_warp, 0.2, 0)
        final = cv2.addWeighted(final, 1, color_warp_lines, 1, 0)

        if self.pub_image_type == 'compressed':
            if self.is_center_x_exist:
                # publishes lane center
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(350)
                self.pub_lane.publish(msg_desired_center)
                self.pre_centerx = msg_desired_center.data

            else:
                msg_desired_center = Float64()
                msg_desired_center.data = self.pre_centerx
                self.pub_lane.publish(msg_desired_center)


            self.pub_image_lane.publish(self.cvBridge.cv2_to_compressed_imgmsg(final, 'jpg'))

        elif self.pub_image_type == 'raw':
            if self.is_center_x_exist:
                # publishes lane center
                msg_desired_center = Float64()
                msg_desired_center.data = centerx.item(350)
                self.pub_lane.publish(msg_desired_center)
                self.pre_centerx = msg_desired_center.data

            else:
                msg_desired_center = Float64()
                msg_desired_center.data = self.pre_centerx
                self.pub_lane.publish(msg_desired_center)

            self.pub_image_lane.publish(self.cvBridge.cv2_to_imgmsg(final, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = DetectLane()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()