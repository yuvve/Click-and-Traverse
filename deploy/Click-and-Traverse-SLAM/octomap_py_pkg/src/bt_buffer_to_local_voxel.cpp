#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <octomap/OcTree.h>
#include <sstream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace py = pybind11;

py::array_t<uint8_t> bt_buffer_to_local_voxel(
    py::bytes bt_data,
    double resolution,
    std::vector<float> robot_pos,   // x, y, z
    float yaw,                      // yaw angle in radians
    int a, int b, int c             // tensor shape in x, y, z
) {
    std::string binary = bt_data;
    std::istringstream stream(binary);

    // Load octree from binary data
    octomap::OcTree tree(resolution);
    if (!tree.readBinary(stream)) {
        throw std::runtime_error("Failed to load octomap from binary.");
    }

    // Create voxel tensor output
    py::array_t<uint8_t> tensor({a, b, c});
    auto buf = tensor.mutable_unchecked<3>();

    // Compute transform: world -> robot facing frame (Z up, yaw around Z)
    Eigen::Affine3f T = Eigen::Affine3f::Identity();
    T.translation() = Eigen::Vector3f(robot_pos[0], robot_pos[1], robot_pos[2]);

    float cos_yaw = std::cos(yaw);
    float sin_yaw = std::sin(yaw);
    Eigen::Matrix3f R;
    R << cos_yaw, -sin_yaw, 0,
         sin_yaw,  cos_yaw, 0,
         0,        0,       1;
    T.linear() = R;

    Eigen::Affine3f T_inv = T.inverse();

    // Traversal
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        if (!tree.isNodeOccupied(*it)) continue;

        Eigen::Vector3f pos_world(it.getX(), it.getY(), it.getZ());
        Eigen::Vector3f pos_local = T_inv * pos_world;

        int ix = static_cast<int>(std::floor(pos_local.x() / resolution + a / 2));
        int iy = static_cast<int>(std::floor(pos_local.y() / resolution + b / 2));
        int iz = static_cast<int>(std::floor(pos_local.z() / resolution + c / 2));

        if (ix >= 0 && ix < a && iy >= 0 && iy < b && iz >= 0 && iz < c) {
            buf(ix, iy, iz) = 1;
        }
    }

    return tensor;
}

PYBIND11_MODULE(my_voxel_lib, m) {
    m.doc() = "Local voxel tensor extractor from Octomap binary";
    m.def("bt_buffer_to_local_voxel", &bt_buffer_to_local_voxel,
          "Convert Octomap binary to local voxel tensor",
          py::arg("bt_data"),
          py::arg("resolution"),
          py::arg("robot_pos"),
          py::arg("yaw"),
          py::arg("a"),
          py::arg("b"),
          py::arg("c"));
}