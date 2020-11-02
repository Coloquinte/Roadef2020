
#include "problem.hpp"
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>

Problem Problem::read(std::istream &is) {
    Problem pb;
    // Read the JSON
    nlohmann::json j;
    is >> j;
    // Initialize the object
    return pb;
}

void Problem::write(std::ostream &os) {
}

Problem Problem::readFile(const std::string &fname) {
    std::ifstream f(fname);
    return read(f);
}

void Problem::writeFile(const std::string &fname) {
    std::ofstream f(fname);
    return write(f);
}

