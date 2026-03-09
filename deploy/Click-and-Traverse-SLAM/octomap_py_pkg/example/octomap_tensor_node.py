import rclpy
from rclpy.node import Node
from octomap_msgs.msg import Octomap
import torch
import octomap_py  # 你自己用 pybind11 编译出的模块

class OctomapTensorNode(Node):
    def __init__(self):
        super().__init__('octomap_tensor_node')

        self.subscription = self.create_subscription(
            Octomap,
            '/octomap_binary',
            self.octomap_callback,
            10
        )

        self.grid_size = 128  # 体素张量维度
        self.get_logger().info("Octomap tensor node initialized.")

    def octomap_callback(self, msg: Octomap):
        self.binary_bt = bytes(msg.data)
        print(self.binary_bt)
        # try:
        #     # 转换为 bytes（msg.data 是 List[int]）
        #     binary_bt = bytes(msg.data)

        #     # 使用 pybind11 模块解析为 numpy → torch.Tensor
        #     voxel_np = octomap_py.bt_buffer_to_voxel(
        #         binary_bt,
        #         resolution=msg.resolution,
        #         grid_size=self.grid_size
        #     )

        #     voxel_tensor = torch.from_numpy(voxel_np).to(torch.uint8)

        #     # 可视化或用于推理
        #     self.get_logger().info(f"Received Octomap → tensor shape: {voxel_tensor.shape}, occupied: {voxel_tensor.sum().item()} voxels")

        # except Exception as e:
        #     self.get_logger().error(f"Failed to parse Octomap: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OctomapTensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
