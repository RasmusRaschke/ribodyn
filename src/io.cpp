#include "io.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <stdexcept>

std::string Input::trim(const std::string& str) const{
    auto first = std::find_if_not(
        str.begin(),
        str.end(),
        [](unsigned char c)
        {
            return std::isspace(c);
        }
    );
    auto last = std::find_if_not(
        str.rbegin(),
        str.rend(),
        [](unsigned char c)
        {
            return std::isspace(c);
        }
    ).base();
    if(first>=last)
        return "";
    return std::string(first, last);
};

bool Input::read(const std::string& filename){
    std::ifstream file(filename);
    if(!file)
        return false;
    std::string line;
    while(std::getline(file,line)){
        auto comment = line.find('#');
        if(comment != std::string::npos)
            line.erase(comment);
        line = trim(line);
        if(line.empty())
            continue;
        std::stringstream ss(line);
        std::string key;
        ss >> key;
        std::string value;
        std::getline(ss, value);
        data[key] = trim(value);
    }
    return true;
};

bool Input::has(const std::string& key) const{
    return data.find(key) != data.end();
};

std::string Input::getString(const std::string& key) const {
    auto it = data.find(key);
    if(it == data.end())
        throw std::runtime_error("Missing input: " + key);
    return it->second;
};

double Input::getDouble(const std::string& key) const{
    return std::stod(getString(key));
};

int Input::getInt(const std::string& key) const {
    return std::stoi(getString(key));
};

bool Input::getBool(const std::string& key) const {
    std::string value = getString(key);
    std::transform(value.begin(), value.end(), value.begin(), ::tolower);
    if(value=="true" || value=="yes" || value=="1" || value=="on")
        return true;
    return false;
};

vec3 Input::getVec3(const std::string& key) const {
    std::stringstream ss(getString(key));
    double x,y,z;
    ss >> x >> y >> z;
    return vec3(x,y,z);
};

Eigen::Vector4d Input::getVec4(const std::string& key) const {
    std::stringstream ss(getString(key));
    double a,b,c,d;
    ss >> a >> b >> c >> d;
    return Eigen::Vector4d(a,b,c,d);
};

mat3 Input::getMat3(const std::string& key) const {
    std::stringstream ss(getString(key));
    mat3 I;
    for (int i=0; i<3; ++i){
        for (int j=0; j<3; j++){
            ss >> I(i,j);
        }
    }
    return I;
}

OutputWriter::OutputWriter(const std::string& filename) : out(filename) {
    if(!out)
        throw std::runtime_error("Cannot open output file.");
};

void OutputWriter::writeHeader()
{
    out
        << "t,"
        << "x,y,z,"
        << "vx,vy,vz,"
        << "qw,qx,qy,qz,"
        << "Ox,Oy,Oz,"
        << "T_trans,T_rot,"
        << "U_generic,U_gr,U_em,E_total,"
        << "Ex_world,Ey_world,Ez_world,"
        << "Ex_body,Ey_body,Ez_body,"
        << "Bx_world,By_world,Bz_world,"
        << "Bx_body,By_body,Bz_body,"
        << "mu_world_x,mu_world_y,mu_world_z,"
        << "mu_body_x,mu_body_y,mu_body_z,"
        << "constraint_residual,"
        << "quaternion_norm\n";
}

void OutputWriter::write(
    double t,
    const State& s,
    const Diagnostics& d
)
{
    out << std::scientific
        << std::setprecision(16);

    out
        << t << ','
        << s.r.x() << ','
        << s.r.y() << ','
        << s.r.z() << ','
        << s.v.x() << ','
        << s.v.y() << ','
        << s.v.z() << ','
        << s.q.w() << ','
        << s.q.x() << ','
        << s.q.y() << ','
        << s.q.z() << ','
        << s.Omega.x() << ','
        << s.Omega.y() << ','
        << s.Omega.z() << ','
        << d.T_trans << ','
        << d.T_rot << ','
        << d.U_generic << ','
        << d.U_gr << ','
        << d.U_em << ','
        << d.E_total << ','
        << d.E_world.x() << ','
        << d.E_world.y() << ','
        << d.E_world.z() << ','
        << d.E_body.x() << ','
        << d.E_body.y() << ','
        << d.E_body.z() << ','
        << d.B_world.x() << ','
        << d.B_world.y() << ','
        << d.B_world.z() << ','
        << d.B_body.x() << ','
        << d.B_body.y() << ','
        << d.B_body.z() << ','
        << d.mu_world.x() << ','
        << d.mu_world.y() << ','
        << d.mu_world.z() << ','
        << d.mu_body.x() << ','
        << d.mu_body.y() << ','
        << d.mu_body.z() << ','
        << d.constraintResidual << ','
        << d.quaternionNorm
        << '\n';
}