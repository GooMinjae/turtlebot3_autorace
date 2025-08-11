import math
import time
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from gazebo_msgs.srv import GetEntityState, SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Quaternion

def yaw_to_q(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q

class PersonMover(Node):
    def __init__(self):
        super().__init__('person_mover')
        self.declare_parameter('name', 'person_model')
        self.declare_parameter('dx', 0.10)       # 100 mm
        self.declare_parameter('wait_sec', 5.0)  # 5 s

        self.name = self.get_parameter('name').get_parameter_value().string_value
        self.dx = float(self.get_parameter('dx').get_parameter_value().double_value)
        self.wait_sec = float(self.get_parameter('wait_sec').get_parameter_value().double_value)

        self.cli_get = self.create_client(GetEntityState, '/gazebo/get_entity_state')
        self.cli_set = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self.get_logger().info('Waiting for Gazebo services (5s timeout each)...')
        if not self.cli_get.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /gazebo/get_entity_state NOT available')
        if not self.cli_set.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('Service /gazebo/set_entity_state NOT available')

        self.timer = self.create_timer(0.1, self.run_once)

    def call_with_timeout(self, future: Future, timeout=3.0):
        start = time.time()
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                return None
        return future.result()

    def get_state(self):
        req = GetEntityState.Request()
        req.name = self.name
        fut = self.cli_get.call_async(req)
        return self.call_with_timeout(fut, timeout=3.0)

    def set_state(self, x, y, z, yaw=None):
        state = EntityState()
        state.name = self.name
        state.pose = Pose()
        state.pose.position.x = float(x)
        state.pose.position.y = float(y)
        state.pose.position.z = float(z)
        if yaw is not None:
            state.pose.orientation = yaw_to_q(float(yaw))
        else:
            state.pose.orientation.w = 1.0
        state.reference_frame = 'world'

        req = SetEntityState.Request()
        req.state = state
        fut = self.cli_set.call_async(req)
        return self.call_with_timeout(fut, timeout=3.0)

    def run_once(self):
        self.timer.cancel()

        res = self.get_state()
        if res is None:
            self.get_logger().error(f'GetEntityState timeout. (서비스/이름/Paused 확인)')
            rclpy.shutdown()
            return
        if not res.success:
            self.get_logger().error(f'GetEntityState failed for name="{self.name}"')
            rclpy.shutdown()
            return

        x0 = res.state.pose.position.x
        y0 = res.state.pose.position.y
        z0 = res.state.pose.position.z
        self.get_logger().info(f'Start pose: x={x0:.3f}, y={y0:.3f}, z={z0:.3f}')

        # +x 이동
        x1 = x0 + self.dx
        self.get_logger().info(f'Move → x={x1:.3f}')
        r = self.set_state(x1, y0, z0, yaw=None)
        if r is None:
            self.get_logger().error('SetEntityState timeout(1)')
        else:
            self.get_logger().info('Moved (+x).')

        # 대기
        self.get_logger().info(f'Wait {self.wait_sec}s')
        time.sleep(self.wait_sec)

        # 180도 회전 + 원위치 복귀
        self.get_logger().info('Turn and go back')
        r2 = self.set_state(x0, y0, z0, yaw=math.pi)
        if r2 is None:
            self.get_logger().error('SetEntityState timeout(2)')
        else:
            self.get_logger().info('Returned.')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    PersonMover()
    # 내부에서 shutdown

if __name__ == '__main__':
    main()

