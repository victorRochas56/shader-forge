#pragma once
#include <glm/glm.hpp>

/*
Colours and metrics for the widgets.
*/
struct GUIStyle {
    glm::vec4 windowBg = glm::vec4(0.10f, 0.11f, 0.14f, 0.92f);
    glm::vec4 titleText = glm::vec4(0.92f, 0.92f, 0.96f, 1.00f);
    glm::vec4 text = glm::vec4(0.86f, 0.87f, 0.90f, 1.00f);
    glm::vec4 textDim = glm::vec4(0.58f, 0.60f, 0.66f, 1.00f);

    glm::vec4 button = glm::vec4(0.24f, 0.26f, 0.32f, 1.00f);
    glm::vec4 buttonHover = glm::vec4(0.32f, 0.35f, 0.43f, 1.00f);
    glm::vec4 buttonActive = glm::vec4(0.38f, 0.42f, 0.52f, 1.00f);
    glm::vec4 buttonOn = glm::vec4(0.26f, 0.55f, 0.36f, 1.00f);
    glm::vec4 buttonOnHover = glm::vec4(0.32f, 0.66f, 0.44f, 1.00f);

    // the trough a slider or a progress bar sits in, and the part of it that is filled
    glm::vec4 frameBg = glm::vec4(0.16f, 0.17f, 0.21f, 1.00f);
    glm::vec4 frameBgHover = glm::vec4(0.20f, 0.22f, 0.27f, 1.00f);
    glm::vec4 sliderFill = glm::vec4(0.28f, 0.48f, 0.78f, 1.00f);
    glm::vec4 sliderFillActive = glm::vec4(0.36f, 0.58f, 0.90f, 1.00f);

    glm::vec4 separator = glm::vec4(1.00f, 1.00f, 1.00f, 0.16f);

    // Popups and tooltips float above everything, so they are opaque — anything showing through
    // reads as part of the window underneath.
    glm::vec4 popupBg = glm::vec4(0.13f, 0.14f, 0.18f, 0.98f);
    glm::vec4 tooltipBg = glm::vec4(0.06f, 0.06f, 0.08f, 0.96f);
    glm::vec4 caret = glm::vec4(0.90f, 0.92f, 0.96f, 1.00f);
    glm::vec4 hintText = glm::vec4(0.48f, 0.50f, 0.56f, 1.00f);

    // seconds the cursor must rest on an item before its tooltip appears
    float tooltipDelay = 0.45f;
    // a popup taller than this scrolls instead of running off the screen
    float popupMaxHeight = 260.0f;
    // seconds per caret blink phase
    float caretBlink = 0.53f;

    glm::vec4 titleBar = glm::vec4(0.17f, 0.19f, 0.25f, 1.00f);
    // the focused window's bar, so which window owns the keyboard reads at a glance
    glm::vec4 titleBarActive = glm::vec4(0.22f, 0.30f, 0.44f, 1.00f);
    glm::vec4 closeHover = glm::vec4(0.72f, 0.26f, 0.26f, 1.00f);

    // px of vertical breathing room inside a framed widget, above and below its text
    float framePaddingY = 3.0f;
    // default width of a slider or an input when setNextItemWidth hasn't said otherwise
    float itemWidth = 140.0f;
};
