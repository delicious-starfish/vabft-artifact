#ifndef EXAMPLES_COMMON_GOLDEN_IO_DATA_HPP
#define EXAMPLES_COMMON_GOLDEN_IO_DATA_HPP

#include <cmath>
#include <vector>
#include <string>
#include <cstdio>

#include <iostream>
#include <fstream>
#include <filesystem> // C++17 支持
#include <vector>

namespace Catlass::golden {

template<class Element>
void saveVectorToBinaryFile(const std::string& filename, const std::vector<Element>& data) {

    // 创建父目录
    std::filesystem::path filePath(filename);
    std::filesystem::path parentPath = filePath.parent_path();
    if (!parentPath.empty() && !std::filesystem::exists(parentPath)) {
        // 递归创建目录
        try {
            std::filesystem::create_directories(parentPath);
        } catch(const std::filesystem::filesystem_error& e) {
            std::cerr << "Error creating directory: " << e.what() << std::endl;
            return;
        }
    }

    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    size_t size = data.size();
    outFile.write(reinterpret_cast<const char*>(&size), sizeof(size)); // Write the size of the vector
    outFile.write(reinterpret_cast<const char*>(data.data()), size * sizeof(Element)); // Write the actual data

    outFile.close();
}

template<class Element>
void loadVectorFromBinaryFile(const std::string& filename, std::vector<Element>& data) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error opening file for reading: " << filename << std::endl;
        return;
    }

    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size)); // Read the size of the vector
    data.resize(size); // Resize the vector to hold the data
    inFile.read(reinterpret_cast<char*>(data.data()), size * sizeof(Element)); // Read the actual data

    inFile.close();
}
}
#endif