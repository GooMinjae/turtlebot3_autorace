import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class SafetyMux(Node):
    def __init__(self):
        super().__init__('safety_cmd_mux')

        # 파라미터: 사람 명령 유효 타임아웃(초)
        self.declare_parameter('person_timeout', 0.3)
        self.person_timeout = float(self.get_parameter('person_timeout').value)

        # 입력: 자동주행, 사람탐지
        self.sub_auto   = self.create_subscription(Twist, '/auto_cmd_vel',   self.on_auto,   10)
        self.sub_person = self.create_subscription(Twist, '/person_cmd_vel', self.on_person, 10)

        # 출력: 실제 로봇에 들어갈 최종 명령
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # 상태
        self.auto_msg = Twist()
        self.person_msg = None
        self.last_person_stamp = self.get_clock().now()

        # 20Hz로 계속 최종 명령 퍼블리시
        self.timer = self.create_timer(0.05, self.tick)

        self.get_logger().info('safety_cmd_mux started: /auto_cmd_vel + /person_cmd_vel -> /cmd_vel')

    def on_auto(self, msg: Twist):
        self.auto_msg = msg

    def on_person(self, msg: Twist):
        self.person_msg = msg
        self.last_person_stamp = self.get_clock().now()

    def tick(self):
        now = self.get_clock().now()
        use_person = False
        if self.person_msg is not None:
            dt = (now - self.last_person_stamp).nanoseconds / 1e9
            use_person = (dt <= self.person_timeout)

        out = self.person_msg if use_person else self.auto_msg
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = SafetyMux()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
