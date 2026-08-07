#include <glm/glm.hpp>
#include <string>

enum GUIAnchor {
    Top =       1 << 0,
    Bottom =    1 << 1,
    Right =     1 << 2,
    Left =      1 << 3,
    Center =    1 << 4
};

enum TextWrap {
    CutOff,
    Wrap,
    FitParent, // make parent fit the text
};

class GUIRect {

public:

    float top() { return center.y + size.y / 2;}
    float bottom() { return center.y - size.y / 2;}
    float left() { return center.x - size.x / 2;}
    float right() { return center.x + size.x / 2;}

private:

    // in px
    glm::vec2 size;
    // in px
    glm::vec2 center;
    glm::vec2 atlasUV;

    GUIAnchor anchor = GUIAnchor::Center;

    // 0 is the sentinel value for empty ref
    uint32_t parent = 0;
    uint32_t nextSibling = 0;
    uint32_t firstChild = 0;


};

// monospace for now
class GUIChar : GUIRect {
    uint32_t charCode;
    uint32_t font;
    uint32_t size;
};

class GUIText : GUIRect {
    // actual text
    std::string content;
    TextWrap wrap = TextWrap::CutOff;
};