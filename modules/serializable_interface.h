#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

class ISerializable {

public:
    const std::string identifier; 

    virtual bool serialize(std::ofstream& ofs) = 0;

    virtual bool parse(std::ifstream& ifs) = 0;
};