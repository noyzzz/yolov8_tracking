#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "pybind11/include/pybind11/numpy.h"
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}



//define a class that has three attributes x, y and height
class Point {
public:
    Point(float x, float y, float depth) : x(x), y(y), depth(depth) {}
    float x, y, depth;
};

//define a map of pair<int, int> to vector<Point>

std::vector<std::vector<float>> depth_cloud2depth_map(std::vector<Point> &depth_cloud, int width, int height, float radius) {
    std::map<std::pair<int, int>, std::vector<Point>> point_map;
    for (uint i = 0; i < depth_cloud.size(); i++) {
        float x = depth_cloud[i].x;
        float y = depth_cloud[i].y;
        for (int j = (int)(x - radius); j < (int)(x + radius); j++) {
            for (int k = (int)(y - radius); k < (int)(y + radius); k++) {
                if (j < 0 || j >= width || k < 0 || k >= height) {
                    continue;
                }
                if (j >= 0 && j < width && k >= 0 && k < height) {
                    point_map[std::make_pair(j, k)].push_back(depth_cloud[i]);
                }
            }
        }
    }
    //initialize the depth map with height of 0.0 with size of width * height
    std::vector<std::vector<float>> depth_map(width, std::vector<float>(height, 0));
    for (auto it = point_map.begin(); it != point_map.end(); it++) {
        int pixel_x = it->first.first;
        int pixel_y = it->first.second;
        float height_sum = 0.0;
        //i want to calculate a weight average of the height of the points in the vector<Point>
        float total_weight = 0.0;
        for (uint i = 0; i < it->second.size(); i++) {
            if (pixel_x == it->second[i].x && pixel_y == it->second[i].y) {
                depth_map[pixel_x][pixel_y] = it->second[i].depth;
                continue;
            }
            float weight = 1.0 / sqrt((pixel_x - it->second[i].x) * (pixel_x - it->second[i].x) + (pixel_y - it->second[i].y) * (pixel_y - it->second[i].y));
            total_weight += weight;
            height_sum += weight * it->second[i].depth;
        }
        depth_map[pixel_x][pixel_y] = height_sum / total_weight;
    }
    return depth_map;
}


 py::array_t<float> depth_cloud2depth_map_wrapper(py::array_t<float> &depth_cloud, int width, int height, float radius) {
    //depth_cloud is a 2D array with shape (n, 3) and n is the number of points and 3 is the x, y and depth. convert it to a vector<Point>
    std::vector<Point> depth_cloud_vector;
    for (int i = 0; i < depth_cloud.shape(0); i++) {
        depth_cloud_vector.push_back(Point(depth_cloud.at(i, 0), depth_cloud.at(i, 1), depth_cloud.at(i, 2)));
    }
    std::vector<std::vector<float>> depth_map = depth_cloud2depth_map(depth_cloud_vector, width, height, radius);
    //convert the depth_map to a 2D numpy array
    py::array_t<float> depth_map_array({width, height});
    auto r = depth_map_array.mutable_unchecked<2>();
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
            r(i, j) = depth_map[i][j];
        }
    }
    return depth_map_array;
    
}

PYBIND11_MODULE(depth_utils, m) {
    m.doc() = "pybind11 depth_utils plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("depth_cloud2depth_map", &depth_cloud2depth_map, "A function that converts depth cloud to depth map");
    m.def("depth_cloud2depth_map_wrapper", &depth_cloud2depth_map_wrapper, "A function that converts depth cloud to depth map");
    py::class_<Point>(m, "Point")
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y)
        .def_readwrite("depth", &Point::depth);
}


