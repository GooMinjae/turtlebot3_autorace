import os
import time
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from geometry_msgs.msg import Pose

class RespawnMover(Node):
    def __init__(self):
        super().__init__('respawn_mover')
        # 파라미터
        self.declare_parameter('name', 'person_model')  # 엔티티 이름
        self.declare_parameter('sdf',  '~/.gazebo/models/person_model/model.sdf')  # SDF 경로
        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)
        self.declare_parameter('start_z', 0.0)
        self.declare_parameter('dx',   0.10)   # 0.10m = 100mm
        self.declare_parameter('wait', 5.0)    # 각 스텝 대기(s)
        self.declare_parameter('cycles', 1)    # 왕복 횟수

        self.name   = self.get_parameter('name').get_parameter_value().string_value
        self.sdf    = os.path.expanduser(self.get_parameter('sdf').get_parameter_value().string_value)
        self.start_x = float(self.get_parameter('start_x').value)
        self.start_y = float(self.get_parameter('start_y').value)
        self.start_z = float(self.get_parameter('start_z').value)
        self.dx     = float(self.get_parameter('dx').value)
        self.wait_s = float(self.get_parameter('wait').value)
        self.cycles = int(self.get_parameter('cycles').value)

        # 서비스 클라이언트 (네임스페이스 없는 기본 서비스 사용)
        self.cli_spawn  = self.create_client(SpawnEntity,  '/spawn_entity')
        self.cli_delete = self.create_client(DeleteEntity, '/delete_entity')

        self.get_logger().info('Waiting for /spawn_entity and /delete_entity ...')
        if not self.cli_spawn.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service /spawn_entity not available'); rclpy.shutdown(); return
        if not self.cli_delete.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Service /delete_entity not available'); rclpy.shutdown(); return
        self.get_logger().info(f'Services ready. target="{self.name}" sdf="{self.sdf}"')

        if not os.path.isfile(self.sdf):
            self.get_logger().error(f'SDF file not found: {self.sdf}'); rclpy.shutdown(); return

        self.timer = self.create_timer(0.2, self.run_once)

    def run_once(self):
        self.timer.cancel()
        for i in range(self.cycles):
            self.get_logger().info(f'=== Cycle {i+1}/{self.cycles} ===')
            # 원점 스폰
            self.safe_delete(self.name)
            self.spawn_at(self.start_x, self.start_y, self.start_z, note='origin')
            time.sleep(self.wait_s)

            # +x로 이동(삭제→재스폰)
            self.safe_delete(self.name)
            self.spawn_at(self.start_x + self.dx, self.start_y, self.start_z, note=f'+x {self.dx:.2f}m')
            time.sleep(self.wait_s)

            # 원점 복귀
            self.safe_delete(self.name)
            self.spawn_at(self.start_x, self.start_y, self.start_z, note='back to origin')
            time.sleep(self.wait_s)

        self.get_logger().info('Done.')
        rclpy.shutdown()

    def safe_delete(self, name):
        req = DeleteEntity.Request(); req.name = name
        fut = self.cli_delete.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        # 삭제 실패해도 계속 진행 (없을 수 있으니까)
        self.get_logger().info(f'DELETE "{name}" (ignored if not exist)')

    def spawn_at(self, x, y, z, note=''):
        pose = Pose(); pose.position.x = float(x); pose.position.y = float(y); pose.position.z = float(z)
        req = SpawnEntity.Request()
        req.name = self.name
        with open(self.sdf, 'r') as f:
            req.xml = f.read()
        req.robot_namespace = ''
        req.reference_frame = 'world'
        req.initial_pose = pose
        fut = self.cli_spawn.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        self.get_logger().info(f'SPAWN "{self.name}" at x={x:.3f}, y={y:.3f}, z={z:.3f} {note}')

def main(args=None):
    rclpy.init(args=args)
    RespawnMover()
    rclpy.spin(rclpy.node.Node('dummy'))

if __name__ == '__main__':
    main()

