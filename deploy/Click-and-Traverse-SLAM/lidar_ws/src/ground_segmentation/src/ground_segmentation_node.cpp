#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl_conversions/pcl_conversions.h>  // 用于转换 PCL 和 ROS 消息

class GroundSegmentationNode : public rclcpp::Node {
public:
    GroundSegmentationNode()
        : Node("ground_segmentation_node") {

        // 创建订阅者，订阅原始点云话题
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/input_point_cloud", 10,
            std::bind(&GroundSegmentationNode::pointCloudCallback, this, std::placeholders::_1));

        // 创建发布者，发布过滤后的点云
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_point_cloud", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // 将 ROS2 消息转换为 PCL 点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        // 创建 RANSAC 分割对象
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);  // 使用平面模型
        seg.setMethodType(pcl::SAC_RANSAC);     // 使用 RANSAC 算法
        seg.setDistanceThreshold(0.02);         // 点到平面的最大距离（单位：米）

        // 存储分割结果
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        seg.setInputCloud(cloud);
        seg.segment(*inliers, *coefficients);

        if (inliers->indices.size() == 0) {
            RCLCPP_ERROR(this->get_logger(), "Could not estimate a planar model for the given dataset.");
            return;
        }

        // 提取地面点（内点）
        pcl::ExtractIndices<pcl::PointXYZ> extract;
        pcl::PointCloud<pcl::PointXYZ>::Ptr ground_cloud(new pcl::PointCloud<pcl::PointXYZ>);
        extract.setInputCloud(cloud);
        extract.setIndices(inliers);
        extract.setNegative(false); // 提取地面点
        extract.filter(*ground_cloud);

        // 将 PCL 点云转换为 ROS2 消息
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*ground_cloud, output_msg);
        output_msg.header = msg->header;  // 保留原始消息的 header

        // 发布过滤后的点云
        pub_->publish(output_msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GroundSegmentationNode>());
    rclcpp::shutdown();
    return 0;
}
