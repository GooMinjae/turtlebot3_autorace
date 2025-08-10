import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
import numpy as np
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from cv_bridge import CvBridge

class YOLONode(Node):
    def __init__(self):
        super().__init__('yolo_sign')
        self.cvBridge = CvBridge()
        self.sub_image_type = 'raw'         # you can choose image type 'compressed', 'raw'
        self.counter = 1

        self.model = YOLO('/home/rokey/Desktop/runs/detect/train2/weights/best.pt')
        self.model.eval()  # YOLOv11 모델 불러오기
        if self.sub_image_type == 'compressed':
            self.sub_image_original = self.create_subscription(
                CompressedImage, '/camera/image_input/compressed', self.get_image, 1)
        else:
            self.sub_image_original = self.create_subscription(
                Image, '/camera/image', self.get_image, 1)

        
        # self.subscriber = self.create_subscription('/camera/image_raw', Image, self.image_callback)
        # self.label_server = self.create_service(YOLO,'/detect/label',self.get_light)
        self.data_publisher = self.create_publisher(String,'/control/label', 1)
        
        self.image_publisher = self.create_publisher(Image,'/detect/yolo_image',  1)
    def get_light(self):
        return self.labels
    
    def get_image(self, image_msg):
        # self.get_logger().info(f'image detected')


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
                self.cv_image = self.cvBridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                self.get_logger().error(f'CvBridge Error: {e}')
                return


                # ROS Image 메시지를 OpenCV 이미지로 변환
        # cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # YOLO 모델을 사용하여 객체 인식
        self.results = self.model(self.cv_image)  # 모델에 이미지 입력

        # 결과를 이미지에 바운딩 박스로 표시
        self.output_image, self.labels = self.draw_boxes(self.cv_image, self.results)
        self.label = String()
        self.label.data = self.labels
        self.data_publisher.publish(self.label)
        # OpenCV 이미지를 ROS Image 메시지로 변환하여 퍼블리시
        self.output_msg = self.cvBridge.cv2_to_imgmsg(self.output_image, encoding="bgr8")
        self.image_publisher.publish(self.output_msg)

        

    def draw_boxes(self, image, results_list):
        label= 'None'
        results = results_list[0]
        labels = [results.names[int(cls)] for cls in results.boxes.cls]
        for label in labels:
            self.get_logger().info(f'label: {label}')

        return image ,label
def main(args=None):
    rclpy.init(args=args)
    node = YOLONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
