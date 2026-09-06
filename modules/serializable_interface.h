#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

// Text helpers for the scene file's `Key : value` blocks. They live here rather than in the loader
// so an ISerializable can read its own block back with exactly the parsing the loader wrote it with.
namespace SerialText {

    inline void trim(std::string& str) {
        size_t start = str.find_first_not_of(" \t\r\n");
        size_t end = str.find_last_not_of(" \t\r\n");
        if (start == std::string::npos || end == std::string::npos) {
            str = "";
        } else {
            str = str.substr(start, end - start + 1);
        }
    }

    inline bool parseKeyValue(const std::string& line, std::string& key, std::string& value) {
        size_t colonPos = line.find(':');
        if (colonPos == std::string::npos) {
            return false;
        }
        key = line.substr(0, colonPos);
        value = line.substr(colonPos + 1);
        trim(key);
        trim(value);
        return true;
    }

    inline std::vector<std::string> split(const std::string& str, char delimiter) {
        std::vector<std::string> result;
        std::stringstream ss(str);
        std::string item;
        while (std::getline(ss, item, delimiter)) {
            trim(item);
            result.push_back(item);
        }
        return result;
    }

}

// Anything that owns state living in the scene file. The loader keeps a list of these and matches
// blocks by `identifier`, so it never has to know what any of them actually are.
class ISerializable {

public:
    // Names this owner's block: serialize() opens `<identifier> {`, and the loader hands back
    // whatever it finds under that name.
    const std::string identifier;

    explicit ISerializable(const std::string& identifier) : identifier(identifier) {}
    virtual ~ISerializable() = default;

    // Writes the whole block, header and closing brace included, into the open scene file.
    virtual bool serialize(std::ofstream& ofs) = 0;

    // Reads one block back. The header line is already consumed, so this starts on the first line
    // inside it and has to stop on its own closing brace — anything left behind is parsed as
    // scene data.
    virtual bool parse(std::ifstream& ifs) = 0;

    // Drops whatever was loaded. Called when the scene is cleared, since state keyed on nodes or
    // templates means nothing once those are gone.
    virtual void clear() {}
};
