#pragma once
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "structures.hpp"


class Input{
    public:
        bool read(const std::string& filename);
        bool has(const std::string& key) const;
        std::string getString(const std::string& key) const;
        double getDouble(const std::string& key) const;
        int getInt(const std::string& key) const;
        bool getBool(const std::string& key) const;
        vec3 getVec3(const std::string& key) const;
        mat3 getMat3(const std::string& key) const;
        Eigen::Vector4d getVec4(const std::string& key) const;

    private:
        std::unordered_map<std::string, std::string> data;
        std::string trim(const std::string&) const;
};

class OutputWriter{
    public:
        OutputWriter(const std::string& filename);
        void writeHeader();
        void write(double t, const State& state, const Diagnostics& diagnostics);
    private:
        std::ofstream out;
};
