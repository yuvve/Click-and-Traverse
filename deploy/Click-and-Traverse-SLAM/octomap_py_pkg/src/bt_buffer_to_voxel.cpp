#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <octomap/OcTree.h>
#include <sstream>

namespace py = pybind11;

py::array_t<uint8_t> bt_buffer_to_voxel(py::bytes data, double resolution, int grid_size) {
    std::string binary = data;
    // std::istringstream stream(binary);
    // std::string header = 
    //     "# Octomap OcTree binary file\n"
    //     "# (feel free to add / change comments, but leave the first line as it is!)\n"
    //     "#\n"
    //     "id OcTree\n"
    //     "res " + std::to_string(resolution) + "\n";

    // // 拼接 header 和 msg.data 内容
    // std::string full_binary = header + binary;

    std::istringstream stream(binary);
    std::cout << "Buffer size: " << binary.size() << " bytes\n";
    // std::cout << binary << "\n";
    octomap::OcTree tree(resolution);
    // if (!tree.readBinary(stream)) {
    //     throw std::runtime_error("Failed to load octomap from buffer.");
    // }
    tree.readBinary(stream);
    // std::cout << tree.size() << " voxels\n";

    auto voxels = py::array_t<uint8_t>({grid_size, grid_size, grid_size});
    std::memset(voxels.mutable_data(), 0, grid_size * grid_size * grid_size * sizeof(uint8_t));

    auto buf = voxels.mutable_unchecked<3>();
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        // std::cout << "voxel\n";
        if (!tree.isNodeOccupied(*it)) continue;

        int gx = static_cast<int>(std::round(it.getX() / resolution)) + grid_size / 2;
        int gy = static_cast<int>(std::round(it.getY() / resolution)) + grid_size / 2;
        int gz = static_cast<int>(std::round(it.getZ() / resolution)) + grid_size / 2;

        if (gx >= 0 && gx < grid_size && gy >= 0 && gy < grid_size && gz >= 0 && gz < grid_size)
            buf(gx, gy, gz) = 1;
        // std::cout << "Occupied voxel: (" << gx << ", " << gy << ", " << gz << ")\n";
    }
    return voxels;
}

PYBIND11_MODULE(octomap_py, m) {
    m.def("bt_buffer_to_voxel", &bt_buffer_to_voxel, "Parse .bt buffer into voxel grid");
}
