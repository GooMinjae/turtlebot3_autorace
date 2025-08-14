#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_rawed', 10)
        self.bridge = CvBridge()

        video_path = '/home/rokey/Desktop/traffic_light1.mp4'  # 여기에 동영상 경로 입력
        self.cap = cv2.VideoCapture(video_path)

        # 30fps 동영상이면 1/30초마다 publish
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().info('Video finished.')
            rclpy.shutdown()
            return

        # OpenCV BGR → ROS Image 메시지
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing video frame')


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
