import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from turtlebot3_msgs.srv import YOLO
from cv_bridge import CvBridge
import torch

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_sign')
        self.bridge = CvBridge()
        self.model = torch.load("yolov11.pt")  # YOLOv11 모델 불러오기
        self.model.eval()
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/detect/image_input/compressed', self.get_image, 1)
        else:
            self.sub_image_original = self.create_subscription(
                Image, '/detect/image_input', self.get_image, 1)

        
        # self.subscriber = self.create_subscription('/camera/image_raw', Image, self.image_callback)
        self.label_server = self.create_service(YOLO,'/detect/label',self.get_light)
        
        self.image_publisher = self.create_publisher('/detect/image', Image, 1)
    def get_light(self):
        return self.labels
    
    def get_image(self, image_msg):

        # Processing every 3 frames to reduce frame processing load
        if self.counter % 3 != 0:
            self.counter += 1
            return
        else:
            self.counter = 1

        if self.sub_image_type == 'compressed':
            np_arr = np.frombuffer(image_msg.data, np.uint8)
            self.cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            try:
                self.cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')
                return

        self.is_image_available = True

                # ROS Image 메시지를 OpenCV 이미지로 변환
        # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YOLO 모델을 사용하여 객체 인식
        self.results = self.model(self.cv_image)  # 모델에 이미지 입력

        # 결과를 이미지에 바운딩 박스로 표시
        self.output_image, self.labels = self.draw_boxes(self.cv_image, self.results)

        # OpenCV 이미지를 ROS Image 메시지로 변환하여 퍼블리시
        self.output_msg = self.bridge.cv2_to_imgmsg(self.output_image, encoding="bgr8")
        self.publisher.publish(self.output_msg)

        

    def draw_boxes(self, image, results):
        # 결과로부터 바운딩 박스를 그리는 함수 (이 부분은 YOLO 모델의 결과 형식에 맞게 수정)
        for result in results.xyxy[0]:  # 결과는 좌표값들로 이루어져 있음
            x1, y1, x2, y2, conf, cls = result
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = results.names[cls]
            labels = []
            labels.append(label)
        return image ,labels
    def main(args=None):
        rclpy.init_node(args=args)
        node = YOLONode()
        rclpy.spin(node)

if __name__ == '__main__':
    pass
