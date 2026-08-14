#pragma once
#include <glm/glm.hpp>
#include <cstdint>


// Auto-layout state for one container element. Pure data.

struct GUILayoutStyle {
    // container edge -> content origin
    glm::vec2 padding = glm::vec2(8.0f, 8.0f);
    // between items: x is what sameLine() inserts, y is the gap between rows
    glm::vec2 itemSpacing = glm::vec2(6.0f, 4.0f);
    // one indent() step, in px
    float indentStep = 18.0f;
    // px of content moved per wheel notch
    float scrollSpeed = 40.0f;
    // Strip at the top of the container the layout must not use — a window's title bar. Kept here
    // rather than folded into padding.y so the scroll extent can tell the two apart.
    float headerHeight = 0.0f;
};

struct GUILayout {
    GUILayoutStyle style;

    // Fixed transparent child covering the container below its header. This is what clips.
    uint32_t viewport = 0;
    // Transparent child of the viewport that everything laid out here is parented to. Scrolling
    // moves it, and it clips nothing. 0 until the first item forces both into existence.
    uint32_t contentRoot = 0;

    // --- cursor, all relative to the content origin (container top-left + padding) ---
    float penX = 0.0f;      // where the next item on this row starts
    float rowTop = 0.0f;    // top of the row being filled
    float rowHeight = 0.0f; // tallest item placed on it so far
    float indent = 0.0f;
    bool rowStarted = false; // false until the first item, so it doesn't advance past an empty row

    // set by sameLine(), consumed by the next placement
    bool sameLinePending = false;
    float sameLineSpacing = -1.0f; // <0 means style.itemSpacing.x

    // set by setNextItemWidth(), consumed by the next placement
    float nextItemWidth = 0.0f; // <=0 means the item keeps its own width

    // furthest right/bottom any item reached, in content space — what auto-sizing reads
    glm::vec2 contentExtent = glm::vec2(0.0f);

    // px the content is displaced by. y is <= 0: scrolling down moves content up.
    glm::vec2 scroll = glm::vec2(0.0f);
    bool scrollable = false;
};
