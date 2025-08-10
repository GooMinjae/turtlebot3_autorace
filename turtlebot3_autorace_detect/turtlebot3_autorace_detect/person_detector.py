import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np

class PersonDetector(Node):
    def __init__(self):
        super().__init__('person_detector')

        # 카메라 이미지 토픽 구독
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_compensated',  # 필요 시 다른 토픽으로 변경 가능
            self.image_callback,
            10
        )

        # 속도 명령 퍼블리셔
        self.pub_human = self.create_publisher(String, '/detect/human', 1)

        # OpenCV 브리지
        self.bridge = CvBridge()

    def image_callback(self, msg):
        try:
            # ROS 이미지 → OpenCV 이미지
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # BGR → HSV 변환
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 핑크색 HSV 범위 설정
            lower_pink = np.array([140, 30, 80])   # 색상 조정됨
            upper_pink = np.array([175, 255, 255])
            mask = cv2.inRange(hsv, lower_pink, upper_pink)

            # 핑크색 영역 픽셀 수
            pink_area = cv2.countNonZero(mask)

            # 로그 출력
            self.get_logger().info(f"핑크색 영역: {pink_area}")

            # 조건에 따라 정지 / 감속 / 주행
            if pink_area >= 1000:
                self.get_logger().info("▶ 사람 감지됨 → 차량 정지")
                for _ in range(10):
                    
                    self.publish_human("Stop")

            elif pink_area >= 700:
                self.get_logger().info("⚠ 사람 근처 감지됨 → 감속 주행")
                self.publish_human("Slow")

            else:
                self.publish_human("NONE")

            # 디버깅용 마스크 시각화
            cv2.imshow("Pink Mask", mask)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"이미지 콜백 오류: {e}")

    def publish_human(self, vel):
        msg = String()
        msg.data = vel
        self.pub_human.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = PersonDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()