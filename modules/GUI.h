#pragma once
#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <queue>

#include "bindless_system.hpp"
#include "constants.hpp"
#include "gui_font.hpp"
#include "gui_layout.hpp"
#include "gui_style.hpp"
#include "render_pass.hpp"
#include "structs.hpp"
#include "input.hpp"

enum GUIAnchor {
    Top =       1 << 0,
    Bottom =    1 << 1,
    Right =     1 << 2,
    Left =      1 << 3,
    Center =    1 << 4
};

// so `Top | Left` stays a GUIAnchor instead of decaying to int
inline GUIAnchor operator|(GUIAnchor a, GUIAnchor b) {
    return static_cast<GUIAnchor>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

enum TextWrap {
    CutOff,
    Wrap,
    FitParent, // make parent fit the text
};

enum GUIType {
    Default =       1 << 0,
    Internal =      1 << 1,
    Button =        1 << 2,
    Resizeable =    1 << 3,
    TextBox =       1 << 4
};

// same reason as GUIAnchor above
inline GUIType operator|(GUIType a, GUIType b) {
    return static_cast<GUIType>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

// `None` is a real bit, not a zero: it's what a rect nobody spawned carries, so handlers can
// select on "not internal" the same way they select on any other subtype.
enum InternalType {
    None =              1 << 0,
    ResizeableHandle =  1 << 1
};

inline InternalType operator|(InternalType a, InternalType b) {
    return static_cast<InternalType>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

class GUI;

// Stands in for IM_ARRAYSIZE at the ported call sites, which pass C arrays of combo item strings.
template <typename T, size_t N>
constexpr int guiArraySize(T (&)[N]) {
    return static_cast<int>(N);
}

// Where window positions/sizes persist between runs
constexpr const char* GUI_LAYOUT_FILE = "gui_layout.ini";

// GUIRect::samplerIndex sentinel: resolve to whatever GUI::init allocated as the default.
constexpr uint32_t GUI_DEFAULT_SAMPLER = 0xFFFFFFFF;
// GUIRect::textureIndex sentinel: the GUI's own font/white atlas.
constexpr uint32_t GUI_ATLAS_TEXTURE = 0xFFFFFFFF;

// Every hook takes the owning GUI and the element's index — handlers stay free functions/lambdas
// with no state of their own. Anything per-element (a drag origin, a caret position) goes in a
// side table keyed by index, not in GUIRect, so GUIRect stays a POD the quad path can memcpy.
using GUIHook = std::function<void(GUI&, uint32_t)>;

/*
Behaviour for one slice of the type space.

Both fields are flag masks ANDed against the element's, and both have to hit for the handler to
run. `type` is by flag rather than equality, so an element typed `Button | Resizeable` runs the
Button handler AND the Resizeable handler, in registration order.

`internalType` is the second axis, and it is not a "don't care" when left alone. It defaults to
None, which is precisely the subtype a rect the user added carries
a plain `{.type = Button}` handler fires on ordinary buttons and NOT on a Resizeable's grips, even though
those are Buttons too. Internal machinery is opt-in: name the subtype to reach it. Either field
can OR several values to widen the match (`None | ResizeableHandle` catches both).
*/
struct GUITypeHandler {
    GUIType type = GUIType::Default;
    InternalType internalType = InternalType::None;

    // After the element is linked into the tree. Spawns internal sub-elements here 
    GUIHook onAdd;
    // Once per frame from handleInput, after hover/pressed/held are resolved for the whole tree.
    GUIHook onInput;
    // Before the element is unlinked, while it and its subtree are still readable. Must not add
    // or remove elements — removeElement's traversal is reentrant-unsafe.
    GUIHook onRemove;
};

/*
Screen-space rect. Pixels, top-left origin, +Y down

Positioning is anchor-relative: `anchor` picks a point on the parent AND the child's own matching
corner, then `offset` displaces the child from there in signed screen px (+x right, +y down).
*/
class GUIRect {

public:

    float top() const { return center.y - size.y / 2; }
    float bottom() const { return center.y + size.y / 2; }
    float left() const { return center.x - size.x / 2; }
    float right() const { return center.x + size.x / 2; }

    void markFree() { self = 0; }
    bool alive() const { return self != 0; }

    GUIType type = GUIType::Default;
    // Which built-in spawned this rect, or None for one the caller added.
    InternalType internalType = InternalType::None;

    // in px
    glm::vec2 size = glm::vec2(0);
    // in px, from the anchor point. authored.
    glm::vec2 offset = glm::vec2(0);
    // in px, absolute. resolved by layout, do not set.
    glm::vec2 center = glm::vec2(0);
    // Absolute px box this rect is trimmed to
    glm::vec2 clipMin = glm::vec2(-1e6f);
    glm::vec2 clipMax = glm::vec2(1e6f);
    // sub-rect of the GUI atlas. Defaults to the atlas' reserved white strip.
    glm::vec2 uvMin = glm::vec2(GUI_WHITE_UV_LO);
    glm::vec2 uvMax = glm::vec2(GUI_WHITE_UV_HI);
    // Any bindless texture, or GUI_ATLAS_TEXTURE for the GUI's own atlas. 
    uint32_t textureIndex = GUI_ATLAS_TEXTURE;
    // Bindless sampler for this rect, or GUI_DEFAULT_SAMPLER to take the GUI's (Nearest/clamp).
    // See GUI::getLinearSampler for the case that wants the other one.
    uint32_t samplerIndex = GUI_DEFAULT_SAMPLER;
    glm::vec4 color = glm::vec4(1);
    glm::vec4 hoverColor = glm::vec4(1, 0, 0, 1);
    glm::vec4 pressedColor = glm::vec4(0, 1, 0, 1);
    glm::vec4 heldColor = glm::vec4(0, 0, 1, 1);

    // Decoration opts out of the hit test with this. A slider's fill bar sits on top of its track
    // and would otherwise take the press the track needs to follow the drag.
    bool hitTestable = true;

    /*
    Whether this rect trims its children to its own bounds, on top of what it inherited.

    That is what a scrolled content root needs: it moves with the scroll offset,
    so clipping to its own bounds would drag the visible region around with it instead of leaving it fixed to the window.
    */
    bool clipsChildren = true;

    GUIAnchor anchor = GUIAnchor::Center;

    // set by hitTest, consumed by uploadGPU. `color` stays the authored value either way,
    // so nothing has to remember how to undo a hover.
    bool hovered = false;
    bool pressed = false;
    bool held = false;

    // 0 is the sentinel value for empty ref
    uint32_t self = 0;
    uint32_t parent = 0;
    uint32_t nextSibling = 0;
    uint32_t firstChild = 0;
    // tail pointer so addElement can append in O(1)
    uint32_t lastChild = 0;
};

/*
One immediate-mode widget's persistent state: which retained element stands in for it, and when it was last drawn.

`lastFrame` is the reconciliation key, GUI::endFrame retires every slot that this frame's calls didn't touch
*/
struct GUIItemSlot {
    uint32_t element = 0;
    uint64_t lastFrame = 0;
    // set while a drag changes the value, read and cleared by isItemDeactivatedAfterEdit
    bool editedWhileActive = false;
    // collapsing header / tree node. The caller keeps no bool of its own
    bool open = false;
    // stops a DefaultOpen flag re-opening a node the user closed on every subsequent frame.
    bool openInitialised = false;
    // slider/drag double-clicked into typing mode: the slot draws a text field over itself
    bool editing = false;
    // nothing typed into that field yet, so the first character replaces the seeded value
    bool editFresh = false;
};


/*
Per-window opt-outs, passed to beginWindow.
*/
enum GUIWindowFlags : uint32_t {
    GUIWindowNone = 0,
    GUIWindowNoMove = 1 << 0,          // the title bar stops dragging the panel
    GUIWindowNoResize = 1 << 1,        // no corner grips
    GUIWindowNoCollapse = 1 << 2,      // no collapse toggle
    GUIWindowNoTitleBar = 1 << 3,      // no chrome at all; implies NoMove and NoCollapse
    GUIWindowNoScroll = 1 << 4,        // the wheel does not move its content
    GUIWindowNoSavedSettings = 1 << 5, // never written to, or read from, the layout file
    // A panel that stays exactly where the caller put it.
    GUIWindowFixed = GUIWindowNoMove | GUIWindowNoResize,
};

inline GUIWindowFlags operator|(GUIWindowFlags a, GUIWindowFlags b) {
    return static_cast<GUIWindowFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

struct GUIWindowState {
    std::string name;
    GUIWindowFlags flags = GUIWindowFlags::GUIWindowNone;
    glm::vec2 offset = glm::vec2(0);
    glm::vec2 size = glm::vec2(0);
    bool collapsed = false;
    // height to come back to when uncollapsed
    float restoreHeight = 0.0f;
};

// Matches the subset of ImGuiTreeNodeFlags the call sites being replaced actually pass.
enum GUITreeFlags : uint32_t {
    TreeNone = 0,
    TreeSelected = 1 << 0,
    TreeLeaf = 1 << 1,
    TreeDefaultOpen = 1 << 2,
};

inline GUITreeFlags operator|(GUITreeFlags a, GUITreeFlags b) {
    return static_cast<GUITreeFlags>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

struct GUITextRun {
    std::string text;
    glm::vec4 color = glm::vec4(1);
    // px inset from the element's top-left, where the first line's box starts
    glm::vec2 padding = glm::vec2(0);
    // currently only cutoff works TODO : expand this to correctly wrapping text and horizontal scrolling
    TextWrap wrap = TextWrap::CutOff;
};

// max GUI Rects
constexpr uint32_t MAX_GUI_ELEMENTS = 8192;
// max of actual uploaded quads to the GPU
constexpr uint32_t MAX_GUI_QUADS = 65536;

// Corner grips a Resizeable spawns for itself. The floor is four grips wide 
// so opposite corners can't meet and swap which edge you think you're dragging.
constexpr float GUI_RESIZE_GRIP_PX = 12.0f;
constexpr float GUI_MIN_RESIZE_PX = 4 * GUI_RESIZE_GRIP_PX;

/*
Retained tree of screen-space rects, flattened to instanced quads once per frame.

Element 0 is both the dead sentinel (self == 0, so it never draws) and the screen rect: layout
stamps it with the current resolution each frame and every root hangs off it.
*/
class GUI {

public:

    void init(BindlessSystem& bindless, GpuContext& gpu) {
        GPUBufferIndex = bindless.descriptorSet->createFixedBuffer<GPUGuiQuad>(MAX_FRAMES_IN_FLIGHT * MAX_GUI_QUADS, true, "GUI quads");
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            bindless.descriptorSet->setBufferFrameOffset(GPUBufferIndex, i, MAX_GUI_QUADS * i);
        }

        std::string fontPath = font.bakeFirstAvailable(GUIFont::defaultCandidates(), GUI_DEFAULT_FONT_PX);
        if (fontPath.empty()) {
            // Not fatal: the white strip exists regardless, so the UI still draws, minus text.
            std::fprintf(stderr, "[GUI] no font found (tried fonts/gui.ttf and the Windows fonts) - text disabled\n");
        }
        auto [atlasImage, atlasMemory, atlasView] =
            resource::createTexture(*bindless.resourceCtx, font.atlasPixels(), GUI_ATLAS_PX, GUI_ATLAS_PX, vk::Format::eR8G8B8A8Unorm);
        atlasTextureIndex = bindless.descriptorSet->allocateTexture(std::move(atlasImage), std::move(atlasMemory), std::move(atlasView), "internal/gui_atlas",
                                                                    false, GUI_ATLAS_PX, GUI_ATLAS_PX);
        font.releasePixels(); // ~1 MB, dead once it is on the GPU; the glyph metrics stay

        // Nearest + clamp to prevent bleeding and have crisp UI, the default for every rect.
        nearestSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eNearest, vk::SamplerMipmapMode::eNearest, vk::SamplerAddressMode::eClampToEdge,
                                                                     VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eNever, vk::BorderColor::eFloatOpaqueBlack);
        // For content that is being rescaled (ie. material thumbnails)
        linearSamplerIndex = bindless.descriptorSet->allocateSampler(vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge,
                                                                    VK_FALSE, 1.0f, VK_FALSE, vk::CompareOp::eNever, vk::BorderColor::eFloatOpaqueBlack);

        // Drawn post-tonemap into the swapchain, so swapchain format, not HDR.
        pipelineIndex = bindless.pipelineManager->createPipeline<GUIPushConstants>(
            PipelineCategory::POSTPROCESS_ALPHA_BLEND, vk::PrimitiveTopology::eTriangleList, vk::CullModeFlagBits::eNone,
            vk::False, vk::False, "shaders/gui.spv",
            bindless.descriptorSet->getDescriptorSetLayout(), bindless.descriptorSet->getDescriptorSet(),
            gpu.getSwapchain().getSwapChainImageFormat());

        registerBuiltinHandlers();
    }

    /*
    Attaches behaviour to a type. Call once per type at startup, before any elements exist — a
    handler registered later will not retroactively fire onAdd for rects that are already in the
    tree. Registering the same type twice runs both. ie:

        gui.registerHandler({.type = GUIType::Button,
                             .onInput = [](GUI& g, uint32_t e) { ... }});

        gui.registerHandler({.type = GUIType::Resizeable,
                             .onAdd = [](GUI& g, uint32_t e) { ... spawn its internals ... },
                             .onInput = [](GUI& g, uint32_t e) { ... },
                             .onRemove = [](GUI& g, uint32_t e) { ... }});

    Naming an internalType narrows the match to the rects some built-in spawned. Omitting it does
    the opposite of a wildcard — see GUITypeHandler.

        gui.registerHandler({.type = GUIType::Internal,
                             .internalType = InternalType::ResizeableHandle,
                             .onInput = [](GUI& g, uint32_t e) { ... }});
    */
    void registerHandler(GUITypeHandler handler) { handlers.push_back(std::move(handler)); }

    /*
    Binds a callback to one element. ie:

        uint32_t btn = gui.addElement(GUIRect{.type = GUIType::Button, ...}, panel);
        gui.setAction(btn, [&renderer](GUI& g, uint32_t e) {
            renderer.features.ssr.enabled = !renderer.features.ssr.enabled;
        });

    Fired by the built-in Button handler on a completed click.
    */
    void setAction(uint32_t index, GUIHook action) { actions[index] = std::move(action); }
    void clearAction(uint32_t index) { actions.erase(index); }

    /*
    Attaches a string to an element. 

    Size is left alone: you can call sizeToText, or set a size yourself and let the clip box trim.
    */
    void setText(uint32_t index, std::string text, glm::vec4 color = glm::vec4(1)) {
        if (get(index) == nullptr) return;
        GUITextRun& run = texts[index];
        run.text = std::move(text);
        run.color = color;
    }

    // printf-style
    void setTextf(uint32_t index, glm::vec4 color, const char* fmt, ...) {
        char buffer[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        setText(index, std::string(buffer), color);
    }

    void clearText(uint32_t index) { texts.erase(index); }

    const GUITextRun* getText(uint32_t index) const {
        auto it = texts.find(index);
        return it == texts.end() ? nullptr : &it->second;
    }

    GUITextRun* getTextMutable(uint32_t index) {
        auto it = texts.find(index);
        return it == texts.end() ? nullptr : &it->second;
    }

    // Grows the element to exactly fit its run, padding included. Only useful once the text is set.
    void sizeToText(uint32_t index) {
        GUIRect* e = get(index);
        const GUITextRun* run = getText(index);
        if (e == nullptr || run == nullptr) return;
        e->size = font.measure(run->text) + run->padding * 2.0f;
    }

    // A label: an element sized to its string and invisible behind it. Transparent in all four
    // states, because a label is its glyphs and nothing else.
    uint32_t addText(const std::string& text, uint32_t parent = 0, GUIAnchor anchor = GUIAnchor::Center,
                     glm::vec2 offset = glm::vec2(0), glm::vec4 color = glm::vec4(1)) {
        uint32_t index = addElement(GUIRect{.offset = offset,
                                            .color = glm::vec4(0),
                                            .hoverColor = glm::vec4(0),
                                            .pressedColor = glm::vec4(0),
                                            .heldColor = glm::vec4(0),
                                            .anchor = anchor},
                                    parent);
        if (index == 0) return 0;
        setText(index, text, color);
        sizeToText(index);
        return index;
    }

    // ===== layout =====

    /*
    Gives a container an auto-layout cursor.
    items added through addToLayout stack downwards
    */
    void beginLayout(uint32_t container, GUILayoutStyle style = {}) {
        if (get(container) == nullptr) return;
        GUILayout& layout = layouts[container];
        layout.style = style;
        resetLayoutCursor(layout);
    }

    // Rewinds the cursor to the top-left without touching the elements already placed
    void resetLayout(uint32_t container) {
        auto it = layouts.find(container);
        if (it != layouts.end()) resetLayoutCursor(it->second);
    }

    GUILayout* getLayout(uint32_t container) {
        auto it = layouts.find(container);
        return it == layouts.end() ? nullptr : &it->second;
    }

    //Adds an element at the container's layout cursor and advances it
    uint32_t addToLayout(GUIRect element, uint32_t container) {
        GUILayout* layout = getLayout(container);
        if (layout == nullptr || get(container) == nullptr) return 0;

        uint32_t root = ensureContentRoot(container, *layout);
        if (root == 0) return 0;

        // setNextItemWidth wins over the rect's own width, and is consumed either way
        if (layout->nextItemWidth > 0.0f) element.size.x = layout->nextItemWidth;
        layout->nextItemWidth = 0.0f;

        element.anchor = GUIAnchor::Top | GUIAnchor::Left;
        element.offset = advanceCursor(*layout, element.size);
        return addElement(element, root);
    }

    // A label placed in the flow, sized to its string.
    uint32_t addTextToLayout(const std::string& text, uint32_t container, glm::vec4 color = glm::vec4(1)) {
        GUIRect rect{.size = measureText(text), .color = glm::vec4(0), .hoverColor = glm::vec4(0),
                     .pressedColor = glm::vec4(0), .heldColor = glm::vec4(0)};
        uint32_t index = addToLayout(rect, container);
        if (index != 0) setText(index, text, color);
        return index;
    }

    // Puts the NEXT item to the right of the one just placed
    void sameLine(uint32_t container, float spacing = -1.0f) {
        GUILayout* layout = getLayout(container);
        if (layout == nullptr || !layout->rowStarted) return;
        layout->sameLinePending = true;
        layout->sameLineSpacing = spacing;
    }

    // Overrides the width of the next item only.
    void setNextItemWidth(uint32_t container, float width) {
        if (GUILayout* layout = getLayout(container)) layout->nextItemWidth = width;
    }

    void indent(uint32_t container, float px = -1.0f) {
        GUILayout* layout = getLayout(container);
        if (layout == nullptr) return;
        layout->indent += px < 0.0f ? layout->style.indentStep : px;
    }

    void unindent(uint32_t container, float px = -1.0f) {
        GUILayout* layout = getLayout(container);
        if (layout == nullptr) return;
        layout->indent = std::max(0.0f, layout->indent - (px < 0.0f ? layout->style.indentStep : px));
    }

    // Full-width rule across the container's content area.
    uint32_t separator(uint32_t container, glm::vec4 color = glm::vec4(1, 1, 1, 0.18f)) {
        GUILayout* layout = getLayout(container);
        GUIRect* c = get(container);
        if (layout == nullptr || c == nullptr) return 0;
        float width = std::max(1.0f, c->size.x - layout->style.padding.x * 2.0f - layout->indent);
        return addToLayout(GUIRect{.size = glm::vec2(width, 1.0f), .color = color, .hoverColor = color,
                                   .pressedColor = color, .heldColor = color},
                           container);
    }

    // A label with a rule filling the rest of the row
    void separatorText(uint32_t container, const std::string& text, glm::vec4 color = glm::vec4(1)) {
        GUILayout* layout = getLayout(container);
        GUIRect* c = get(container);
        if (layout == nullptr || c == nullptr) return;

        glm::vec2 textSize = measureText(text);
        addTextToLayout(text, container, color);

        float remaining = c->size.x - layout->style.padding.x * 2.0f - layout->indent - textSize.x - layout->style.itemSpacing.x * 2.0f;
        if (remaining < 4.0f) return; // no room for a rule worth drawing

        sameLine(container);
        glm::vec4 lineColor(color.r, color.g, color.b, 0.18f);
        uint32_t rule = addToLayout(GUIRect{.size = glm::vec2(remaining, 1.0f), .color = lineColor, .hoverColor = lineColor,
                                            .pressedColor = lineColor, .heldColor = lineColor},
                                    container);
        // centre the rule against the text's midline rather than sitting it on the text's top edge
        if (GUIRect* r = get(rule)) r->offset.y += std::round(textSize.y * 0.5f);
    }

    // Grows the container to exactly contain what has been laid out in it
    void fitToContent(uint32_t container) {
        GUILayout* layout = getLayout(container);
        GUIRect* c = get(container);
        if (layout == nullptr || c == nullptr) return;
        c->size = layout->contentExtent + layout->style.padding * 2.0f;
        c->size.y += layout->style.headerHeight; // the title bar is not content space
    }

    // px of content that does not fit the container, per axis, 0 when it all fits
    glm::vec2 getScrollableOverflow(uint32_t container) {
        GUILayout* layout = getLayout(container);
        GUIRect* c = get(container);
        if (layout == nullptr || c == nullptr) return glm::vec2(0);
        return overflowOf(*layout, *c);
    }

    // Opt a container into wheel scrolling. Only meaningful once its content overflows.
    void setScrollable(uint32_t container, bool scrollable) {
        if (GUILayout* layout = getLayout(container)) layout->scrollable = scrollable;
    }

    // Opens a frame. Every widget call belongs between this and endFrame.    
    void beginFrame() {
        #if TRACY_ENABLE
        ZoneScopedN("gui beginFrame");
        #endif
        frameCounter++;
        frameTime = std::chrono::steady_clock::now();
        idStack.clear();
        windowStack.clear();
        autoIDStack.clear();
        comboStack.clear();
        lastItem = 0;
    }


    // Closes the frame and retires anything the caller stopped drawing.    
    void endFrame() {
        #if TRACY_ENABLE
        ZoneScopedN("gui endFrame"); // the retire sweep — scales with live item count
        #endif
        // before the sweep, so the tooltip element exists and is marked alive for this frame
        flushTooltip();
        pendingTooltip.clear();

        for (auto it = imItems.begin(); it != imItems.end();) {
            // Either untouched this frame, or already dead because an ancestor was retired first
            // and took the subtree with it.
            if (it->second.lastFrame != frameCounter || get(it->second.element) == nullptr) {
                removeElement(it->second.element);
                dragAccum.erase(it->first);
                sliderAccum.erase(it->first);
                caretPos.erase(it->first);
                inputBuffers.erase(it->first);
                it = imItems.erase(it);
            } else {
                ++it;
            }
        }
    }

    /*
    Opens a window, creating it at defaultPos/defaultSize the first time it is seen. 
    Returns false if it could not be opened.
    */
    bool beginWindow(const char* label, bool* open = nullptr, glm::vec2 defaultSize = glm::vec2(300, 220),
                     glm::vec2 defaultPos = glm::vec2(20, 20), GUIAnchor anchor = GUIAnchor::Top | GUIAnchor::Left,
                     GUIWindowFlags flags = GUIWindowFlags::GUIWindowNone) {
        // Closed windows are not touched, so the sweep retires the panel.
        if (open != nullptr && !*open) return false;

        uint64_t id = hashID(label, currentID());
        std::string title(displayPart(label));
        GUIWindowState& state = windowStates[id];
        state.flags = flags;
        if (state.name.empty()) {
            state.name = title;
            // A layout loaded from disk wins over the caller's defaults, but only the first time,
            // and never for a window that opted out of persistence.
            auto saved = savedLayout.find(title);
            bool restore = saved != savedLayout.end() && !(flags & GUIWindowNoSavedSettings);
            state.offset = restore ? saved->second.offset : defaultPos;
            state.size = restore ? saved->second.size : defaultSize;
            state.collapsed = restore && saved->second.collapsed;
            state.restoreHeight = state.size.y;
        }

        // setNextWindowPos overrides the stored offset
        if (hasNextWindowPos) {
            state.offset = nextWindowPos;
            hasNextWindowPos = false;
            if (GUIRect* existing = get(imItems[id].element)) existing->offset = nextWindowPos;
        }

        // setNextWindowSize overrides the stored size. restoreHeight follows, or a collapse/expand
        // would snap back to the height this window happened to have before.
        if (hasNextWindowSize) {
            state.size = nextWindowSize;
            state.restoreHeight = nextWindowSize.y;
            hasNextWindowSize = false;
            // drawWindowChrome runs after this and re-applies the collapsed height, so a collapsed
            // window keeps its title bar and takes the new size on expand.
            if (GUIRect* existing = get(imItems[id].element)) existing->size = nextWindowSize;
        }

        bool hasTitleBar = !(flags & GUIWindowNoTitleBar);

        GUIItemSlot& slot = imItems[id];
        slot.lastFrame = frameCounter;
        if (get(slot.element) == nullptr) {
            // Resizeable is what grows the corner grips, so NoResize is expressed by not being one.
            slot.element = addElement(GUIRect{.type = (flags & GUIWindowNoResize) ? GUIType::Default : GUIType::Resizeable,
                                              .size = state.size,
                                              .offset = state.offset,
                                              .color = style.windowBg,
                                              .hoverColor = style.windowBg,
                                              .pressedColor = style.windowBg,
                                              .heldColor = style.windowBg,
                                              .anchor = anchor});
            if (slot.element == 0) return false;
            beginLayout(slot.element, GUILayoutStyle{.padding = glm::vec2(8.0f, 6.0f),
                                                     .headerHeight = hasTitleBar ? titleBarHeight() : 0.0f});
            setScrollable(slot.element, !(flags & GUIWindowNoScroll));
        }
        uint32_t panel = slot.element;

        bool collapsed = hasTitleBar ? drawWindowChrome(panel, id, title, open) : false;
        // the grips resize it, the title bar moves it — either way the persisted state has to
        // follow, or a close/reopen (or a restart) would snap it back
        if (GUIRect* p = get(panel)) {
            windowStates[id].offset = p->offset;
            windowStates[id].size = glm::vec2(p->size.x, windowStates[id].restoreHeight);
        }
        windowStates[id].collapsed = collapsed;
        if (collapsed) return false; // caller skips the body AND endWindow — see the contract below

        // The cursor rewinds every frame — that is the whole trick. Items then re-place themselves
        // in call order, so a widget that disappears closes the gap it left behind.
        resetLayout(panel);
        pushWindowScope(panel, id);
        return true;
    }

    // Closes the window opened by a beginWindow that returned true.
    void endWindow() { popWindowScope(); }

    float titleBarHeight() const { return font.lineHeight() + style.framePaddingY * 2.0f; }

    // ----- layout persistence -----

    // Positions, sizes and collapsed flags for every window seen this session, by name.
    void saveLayout(const std::string& path) const {
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "wb") != 0 || f == nullptr) return;
        fprintf(f, "# shader-forge gui layout\n");
        for (const auto& [id, state] : windowStates) {
            if (state.name.empty() || (state.flags & GUIWindowNoSavedSettings)) continue;
            // tab-separated so a window name may contain spaces
            fprintf(f, "%s\t%.1f %.1f %.1f %.1f %d\n", state.name.c_str(), state.offset.x, state.offset.y, state.size.x,
                    state.size.y, state.collapsed ? 1 : 0);
        }
        fclose(f);
    }

    // Call before the first frame
    void loadLayout(const std::string& path) {
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") != 0 || f == nullptr) return;
        char line[512];
        while (fgets(line, sizeof(line), f) != nullptr) {
            if (line[0] == '#') continue;
            char* tab = strchr(line, '\t');
            if (tab == nullptr) continue;
            *tab = '\0';

            GUIWindowState state;
            state.name = line;
            int collapsed = 0;
            if (sscanf_s(tab + 1, "%f %f %f %f %d", &state.offset.x, &state.offset.y, &state.size.x, &state.size.y,
                         &collapsed) != 5) {
                continue;
            }
            state.collapsed = collapsed != 0;
            state.restoreHeight = state.size.y;
            savedLayout[state.name] = state;
        }
        fclose(f);
    }

    // The element the current window is; 0 outside any window.
    uint32_t currentWindow() const { return windowStack.empty() ? 0 : windowStack.back(); }

    /*
    Widens the id scope. Two widgets with the same label in different scopes are different widgets,
    which is what the `##<index>` suffixes in the ImGui call sites are doing by hand — and those
    port straight across, since the hash takes the whole label including the suffix.
    */
    void pushID(std::string_view label) { idStack.push_back(hashID(label, currentID())); }
    void pushID(int i) { idStack.push_back(hashID(std::string_view(reinterpret_cast<const char*>(&i), sizeof(i)), currentID())); }
    void popID() {
        if (!idStack.empty()) idStack.pop_back();
    }

    void sameLine() { sameLine(currentWindow(), -1.0f); }
    void setNextItemWidth(float width) { setNextItemWidth(currentWindow(), width); }
    void indent() { indent(currentWindow(), -1.0f); }
    void unindent() { unindent(currentWindow(), -1.0f); }

    // Full-width rule across the current window's content.
    void separator() {
        uint32_t window = currentWindow();
        GUILayout* layout = getLayout(window);
        GUIRect* c = get(window);
        if (layout == nullptr || c == nullptr) return;

        float width = std::max(1.0f, c->size.x - layout->style.padding.x * 2.0f - layout->indent);
        acquireItem(nextAutoID("##sep"), GUIRect{.size = glm::vec2(width, 1.0f),
                                                  .color = style.separator,
                                                  .hoverColor = style.separator,
                                                  .pressedColor = style.separator,
                                                  .heldColor = style.separator,
                                                  .hitTestable = false});
    }

    // A label with a rule filling the rest of the row.
    void separatorText(std::string_view label) {
        uint32_t window = currentWindow();
        GUILayout* layout = getLayout(window);
        GUIRect* c = get(window);
        if (layout == nullptr || c == nullptr) return;

        std::string_view display = displayPart(label);
        glm::vec2 textSize = measureText(display);
        textColored(label, style.textDim);

        float remaining = c->size.x - layout->style.padding.x * 2.0f - layout->indent - textSize.x - layout->style.itemSpacing.x * 2.0f;
        if (remaining < 4.0f) return; // no room for a rule worth drawing

        sameLine(window);
        uint32_t rule = acquireItem(nextAutoID("##septext"), GUIRect{.size = glm::vec2(remaining, 1.0f),
                                                                      .color = style.separator,
                                                                      .hoverColor = style.separator,
                                                                      .pressedColor = style.separator,
                                                                      .heldColor = style.separator,
                                                                      .hitTestable = false});
        // sit the rule on the text's midline rather than its top edge
        if (GUIRect* r = get(rule)) r->offset.y += std::round(textSize.y * 0.5f);
    }

    // Plain label in the current window's flow.
    void text(std::string_view label) { textColored(label, style.text); }

    // Identified by position in the call sequence
    void textColored(std::string_view label, glm::vec4 color) {
        std::string_view display = displayPart(label);
        uint32_t item = acquireItem(nextAutoID("##text"), GUIRect{.size = measureText(display),
                                                                   .color = glm::vec4(0),
                                                                   .hoverColor = glm::vec4(0),
                                                                   .pressedColor = glm::vec4(0),
                                                                   .heldColor = glm::vec4(0),
                                                                   .hitTestable = false});
        if (item != 0) setText(item, std::string(display), color);
    }

    // printf-style, the ~20 formatted call sites' direct replacement.
    void textf(const char* fmt, ...) {
        char buffer[512];
        va_list args;
        va_start(args, fmt);
        vsnprintf(buffer, sizeof(buffer), fmt, args);
        va_end(args);
        textColored(buffer, style.text);
    }

    // ImGui::ColorConvertHSVtoRGB, needed by the per-scope trace colouring.
    static glm::vec3 hsvToRgb(float h, float s, float v) {
        if (s <= 0.0f) return glm::vec3(v);
        h = std::fmod(h, 1.0f) * 6.0f;
        int sector = static_cast<int>(h);
        float f = h - sector;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));
        switch (sector) {
            case 0: return glm::vec3(v, t, p);
            case 1: return glm::vec3(q, v, p);
            case 2: return glm::vec3(p, v, t);
            case 3: return glm::vec3(p, q, v);
            case 4: return glm::vec3(t, p, v);
            default: return glm::vec3(v, p, q);
        }
    }

    // True on the frame the button is clicked (press and release both inside it).
    bool button(std::string_view label, glm::vec2 size = glm::vec2(0)) {
        return buttonImpl(label, size, style.button, style.buttonHover);
    }

    // A button already drawn in its active colour
    bool buttonToggled(std::string_view label, bool active, glm::vec2 size = glm::vec2(0)) {
        return buttonImpl(label, size, active ? style.buttonOn : style.button,
                          active ? style.buttonOnHover : style.buttonHover);
    }

    // px left on the current row, i.e. what a -1 width resolves to. ImGui's -1 sizing convention.
    float availableWidth() {
        GUILayout* layout = getLayout(currentWindow());
        GUIRect* c = get(currentWindow());
        if (layout == nullptr || c == nullptr) return style.itemWidth;
        return std::max(1.0f, c->size.x - layout->style.padding.x * 2.0f - layout->indent);
    }

    // Positions the next window this frame, overriding its stored offset. ImGui's SetNextWindowPos
    // — the context menu uses it to open at the cursor.
    void setNextWindowPos(glm::vec2 position) {
        nextWindowPos = position;
        hasNextWindowPos = true;
    }

    // Sizes the next window this frame, overriding its stored size — the defaultSize argument only
    // applies the first time a window is seen, so this is the way to drive a size that changes.
    // On a resizeable window, calling it every frame fights the grips.
    void setNextWindowSize(glm::vec2 size) {
        nextWindowSize = size;
        hasNextWindowSize = true;
    }

    // Returns true on the frame it is toggled, and writes through the pointer — ImGui's contract.
    bool checkbox(std::string_view label, bool* value) {
        if (value == nullptr) return false;
        uint64_t id = hashID(label, currentID());
        float box = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(box),
                                                 .color = style.frameBg,
                                                 .hoverColor = style.frameBgHover,
                                                 .pressedColor = style.frameBgHover,
                                                 .heldColor = style.frameBgHover});
        if (item == 0) return false;

        bool toggled = clickedElement == item;
        if (toggled) *value = !*value;

        // the tick is an inset block: no glyph for it in an ASCII atlas, and a filled square reads
        // as unambiguously as a check does at this size
        glm::vec4 tick = *value ? style.sliderFill : glm::vec4(0);
        acquireChild(hashID("##tick", id), item, GUIRect{.size = glm::vec2(box - 8.0f),
                                                          .color = tick,
                                                          .hoverColor = tick,
                                                          .pressedColor = tick,
                                                          .heldColor = tick,
                                                          .hitTestable = false,
                                                          .anchor = GUIAnchor::Center});
        labelAfter(label);
        setLastItem(item, id);
        return toggled;
    }

    // fraction is clamped to 0..1; overlay is drawn centred over the bar, empty for none.
    void progressBar(float fraction, glm::vec2 size = glm::vec2(0), std::string_view overlay = {}) {
        uint64_t id = nextAutoID("##progress");
        if (size.x < 0.0f) size.x = availableWidth();
        else if (size.x == 0.0f) size.x = style.itemWidth;
        if (size.y <= 0.0f) size.y = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t item = acquireItem(id, GUIRect{.size = size,
                                                 .color = style.frameBg,
                                                 .hoverColor = style.frameBg,
                                                 .pressedColor = style.frameBg,
                                                 .heldColor = style.frameBg,
                                                 .hitTestable = false});
        if (item == 0) return;

        float t = glm::clamp(fraction, 0.0f, 1.0f);
        acquireChild(hashID("##fill", id), item, GUIRect{.size = glm::vec2(size.x * t, size.y),
                                                          .color = style.sliderFill,
                                                          .hoverColor = style.sliderFill,
                                                          .pressedColor = style.sliderFill,
                                                          .heldColor = style.sliderFill,
                                                          .hitTestable = false,
                                                          .anchor = GUIAnchor::Top | GUIAnchor::Left});
        // after the fill, for the same reason as the slider's value, text on the bar itself would be painted over by it
        if (!overlay.empty()) captionChild(hashID("##overlay", id), item, size, overlay, style.text);
    }

    // Draws any bindless texture. 
    void image(uint32_t texture, glm::vec2 size, glm::vec2 uvMin = glm::vec2(0), glm::vec2 uvMax = glm::vec2(1)) {
        acquireItem(nextAutoID("##image"), GUIRect{.size = size,
                                                    .uvMin = uvMin,
                                                    .uvMax = uvMax,
                                                    .textureIndex = texture,
                                                    .samplerIndex = linearSamplerIndex,
                                                    .hitTestable = false});
    }

    // Same, but clickable. The label is id-only, nothing is drawn over the image.
    bool imageButton(std::string_view label, uint32_t texture, glm::vec2 size,
                     glm::vec2 uvMin = glm::vec2(0), glm::vec2 uvMax = glm::vec2(1)) {
        uint64_t id = hashID(label, currentID());
        // tint rather than a border: the image fills the whole rect, so hover has to show through it
        uint32_t item = acquireItem(id, GUIRect{.size = size,
                                                 .uvMin = uvMin,
                                                 .uvMax = uvMax,
                                                 .textureIndex = texture,
                                                 .samplerIndex = linearSamplerIndex,
                                                 .color = glm::vec4(1),
                                                 .hoverColor = glm::vec4(1.25f, 1.25f, 1.25f, 1.0f),
                                                 .pressedColor = glm::vec4(0.75f, 0.75f, 0.75f, 1.0f),
                                                 .heldColor = glm::vec4(0.75f, 0.75f, 0.75f, 1.0f)});
        if (item == 0) return false;
        setLastItem(item, id);
        return clickedElement == item;
    }

    /*
    Drag-anywhere-on-the-track slider. Returns true on any frame the value changed.

    Hold shift while dragging to narrow the travel to style.fineDragScale; shift-pressing the track
    also skips the jump-to-cursor, so a fine adjustment starts from the value that is already set.
    Double click to type a number in instead — see numericField.
    */
    bool sliderFloat(std::string_view label, float* value, float min, float max, const char* format = "%.3f", bool samelineLabel = true) {
        if (value == nullptr) return false;

        if(!samelineLabel)
            addLabel(label);

        std::string_view display = displayPart(label);
        uint64_t id = hashID(label, currentID());
        if (editingItem(id)) return numericField(id, label, value, min, max);

        GUILayout* layout = getLayout(currentWindow());
        float width = (layout != nullptr && layout->nextItemWidth > 0.0f) ? layout->nextItemWidth : style.itemWidth;
        float height = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t track = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(width, height),
                                                 .color = style.frameBg,
                                                 .hoverColor = style.frameBgHover,
                                                 .pressedColor = style.frameBgHover,
                                                 .heldColor = style.frameBgHover});
        if (track == 0) return false;

        bool changed = false;
        GUIRect* rect = get(track);
        if (rect != nullptr && max > min) {
            const auto& in = InputManager::getCurrentState();
            float span = std::max(rect->size.x, 1.0f);

            // The press arms the drag, every frame after it moves the value by the cursor's travel.
            // Relative rather than absolute so shift can change the gain mid-drag without the value
            // jumping to wherever the cursor happens to be. Kept clamped, so overshooting an end
            // costs no un-dragging on the way back — which is what the absolute mapping did too.
            if (captured == track && capturedPrev != track) {
                float t = glm::clamp((in.mousePos.x - rect->left()) / span, 0.0f, 1.0f);
                sliderAccum[id] = fineModifierHeld() ? *value : min + t * (max - min);
            }
            if (rect->held) {
                auto accum = sliderAccum.find(id);
                // armed by the press above; only a drag inherited from a frame this slider sat out
                // lands here, and it picks up from wherever the value already is
                if (accum == sliderAccum.end()) accum = sliderAccum.emplace(id, *value).first;

                float gain = fineModifierHeld() ? style.fineDragScale : 1.0f;
                accum->second = glm::clamp(accum->second + (in.mouseDelta.x / span) * (max - min) * gain, min, max);
                if (accum->second != *value) {
                    *value = accum->second;
                    changed = true;
                }
            }
        }
        // Latched, not per-frame: isItemDeactivatedAfterEdit has to answer "did this drag change
        // anything", and the frame the drag ends is never the frame the value moved.
        if (changed) imItems[id].editedWhileActive = true;

        float fraction = max > min ? glm::clamp((*value - min) / (max - min), 0.0f, 1.0f) : 0.0f;
        bool active = rect != nullptr && rect->held;
        glm::vec4 fillColor = active ? style.sliderFillActive : style.sliderFill;
        // A child, so it draws over the track. Not hit-testable, or it would take the press the
        // track needs to follow the drag.
        acquireChild(hashID("##fill", id), track,
                     GUIRect{.size = glm::vec2(width * fraction, height),
                             .color = fillColor,
                             .hoverColor = fillColor,
                             .pressedColor = fillColor,
                             .heldColor = fillColor,
                             .hitTestable = false,
                             .anchor = GUIAnchor::Top | GUIAnchor::Left});

        // after the fill, so the value reads on top of it rather than under it
        char valueText[64];
        snprintf(valueText, sizeof(valueText), format, *value);
        captionChild(hashID("##value", id), track, glm::vec2(width, height), valueText, style.text);
        
        if(samelineLabel)
            labelAfter(label);
        // the caption is an item of its own and moved these; the queries have to answer about the
        // slider, not its label
        setLastItem(track, id);
        if (doubleClickedElement == track) beginNumericEdit(id, track, *value);
        return changed;
    }

    bool sliderInt(std::string_view label, int* value, int min, int max, const char* format = "%d") {
        if (value == nullptr) return false;
        uint64_t id = hashID(label, currentID());
        float asFloat = static_cast<float>(*value);
        // +0.999 so each integer owns an equal slice of the track rather than only its exact point
        bool changed = sliderFloat(label, &asFloat, static_cast<float>(min), static_cast<float>(max) + 0.999f, "");
        int next = glm::clamp(static_cast<int>(asFloat), min, max);
        if (changed && next != *value) {
            *value = next;
        } else {
            changed = false;
        }
        // the typing field draws the characters being typed; a caption would sit on top of them
        if (editingItem(id) || justCommitted(id)) return changed;

        char valueText[64];
        snprintf(valueText, sizeof(valueText), format, *value);
        uint32_t track = itemOf(id);
        GUIRect* rect = get(track);
        if (rect != nullptr) captionChild(hashID("##value", id), track, rect->size, valueText, style.text);
        return changed;
    }

    // Shift narrows the travel to style.fineDragScale; double click to type a number in.
    bool dragFloat(std::string_view label, float* value, float speed = 1.0f, float min = 0.0f, float max = 0.0f,
                   const char* format = "%.3f") {
        if (value == nullptr) return false;
        uint64_t id = hashID(label, currentID());
        if (editingItem(id)) return numericField(id, label, value, min, max);

        GUILayout* layout = getLayout(currentWindow());
        float width = (layout != nullptr && layout->nextItemWidth > 0.0f) ? layout->nextItemWidth : style.itemWidth;
        float height = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(width, height),
                                                 .color = style.frameBg,
                                                 .hoverColor = style.frameBgHover,
                                                 .pressedColor = style.sliderFill,
                                                 .heldColor = style.sliderFill});
        if (item == 0) return false;

        bool changed = false;
        GUIRect* rect = get(item);
        if (rect != nullptr && rect->held) {
            float dx = InputManager::getCurrentState().mouseDelta.x;
            if (dx != 0.0f) {
                *value += dx * speed * (fineModifierHeld() ? style.fineDragScale : 1.0f);
                if (min < max) *value = glm::clamp(*value, min, max);
                changed = true;
            }
        }
        if (changed) imItems[id].editedWhileActive = true;

        char valueText[64];
        snprintf(valueText, sizeof(valueText), format, *value);
        setText(item, valueText, style.text);
        centerText(item);

        labelAfter(label);
        setLastItem(item, id);
        if (doubleClickedElement == item) beginNumericEdit(id, item, *value);
        return changed;
    }

    bool dragInt(std::string_view label, int* value, float speed = 1.0f, int min = 0, int max = 0, const char* format = "%d") {
        if (value == nullptr) return false;
        // Accumulated in float so a sub-pixel-per-frame speed still moves: rounding the delta
        // every frame instead would quantise any speed below 1 straight to zero.
        uint64_t id = hashID(label, currentID());
        float& accum = dragAccum[id];
        GUIRect* prev = get(itemOf(id));
        if (prev == nullptr || !prev->held) accum = static_cast<float>(*value);

        bool changed = dragFloat(label, &accum, speed, static_cast<float>(min), static_cast<float>(max), "");
        int next = static_cast<int>(std::round(accum));
        if (min < max) next = glm::clamp(next, min, max);
        changed = changed && next != *value;
        if (changed) *value = next;
        // while typing, the field owns the element's text — see sliderInt
        if (editingItem(id) || justCommitted(id)) return changed;

        char valueText[64];
        snprintf(valueText, sizeof(valueText), format, *value);
        if (uint32_t item = itemOf(id)) {
            setText(item, valueText, style.text);
            centerText(item);
        }
        return changed;
    }

    // Two drags on one row sharing a label, as ImGui lays them out.
    bool dragFloat2(std::string_view label, float* values, float speed = 1.0f, float min = 0.0f, float max = 0.0f,
                    const char* format = "%.3f") {
        if (values == nullptr) return false;
        pushID(label);
        uint64_t idX = hashID("##x", currentID());
        uint64_t idY = hashID("##y", currentID());
        GUILayout* layout = getLayout(currentWindow());
        float full = (layout != nullptr && layout->nextItemWidth > 0.0f) ? layout->nextItemWidth : style.itemWidth;
        float each = (full - (layout != nullptr ? layout->style.itemSpacing.x : 6.0f)) * 0.5f;

        setNextItemWidth(currentWindow(), each);
        bool changed = dragFloat("##x", &values[0], speed, min, max, format);
        sameLine(currentWindow());
        setNextItemWidth(currentWindow(), each);
        changed |= dragFloat("##y", &values[1], speed, min, max, format);
        popID();

        labelAfter(label);
        // Report on whichever half the user actually touched. isItemDeactivatedAfterEdit after a
        // dragFloat2 has to fire for the end the drag finished on — the particle emitter re-reserves
        // its pool on exactly that edge, and reporting on a fixed half would miss half the drags.
        uint64_t focus = subItemInFocus(idX, idY);
        setLastItem(itemOf(focus), focus);
        return changed;
    }

    // Two ends of a range that cannot cross each other.
    // formatMax defaults to `format`
    bool dragFloatRange2(std::string_view label, float* currentMin, float* currentMax, float speed = 1.0f,
                         float min = 0.0f, float max = 0.0f, const char* format = "%.3f",
                         const char* formatMax = nullptr) {
        if (currentMin == nullptr || currentMax == nullptr) return false;
        pushID(label);
        uint64_t idLo = hashID("##min", currentID());
        uint64_t idHi = hashID("##max", currentID());
        GUILayout* layout = getLayout(currentWindow());
        float full = (layout != nullptr && layout->nextItemWidth > 0.0f) ? layout->nextItemWidth : style.itemWidth;
        float each = (full - (layout != nullptr ? layout->style.itemSpacing.x : 6.0f)) * 0.5f;

        setNextItemWidth(currentWindow(), each);
        bool changed = dragFloat("##min", currentMin, speed, min, max, format);
        sameLine(currentWindow());
        setNextItemWidth(currentWindow(), each);
        changed |= dragFloat("##max", currentMax, speed, min, max, formatMax != nullptr ? formatMax : format);
        popID();

        // clamp against each other rather than swapping: swapping mid-drag hands the drag to the
        // other end and the range jumps
        if (*currentMin > *currentMax) {
            if (changed) *currentMax = std::max(*currentMax, *currentMin);
            *currentMin = std::min(*currentMin, *currentMax);
        }

        labelAfter(label);
        uint64_t focus = subItemInFocus(idLo, idHi);
        setLastItem(itemOf(focus), focus);
        return changed;
    }

    // Full-width clickable row. True on the frame it is clicked; `selected` only drives the paint.
    bool selectable(std::string_view label, bool selected = false) {
        uint64_t id = hashID(label, currentID());
        uint32_t window = currentWindow();
        GUILayout* layout = getLayout(window);
        GUIRect* c = get(window);
        if (layout == nullptr || c == nullptr) return false;

        float width = std::max(1.0f, c->size.x - layout->style.padding.x * 2.0f - layout->indent);
        glm::vec4 base = selected ? style.buttonActive : glm::vec4(0);
        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(width, getLineHeight() + style.framePaddingY),
                                                 .color = base,
                                                 .hoverColor = style.buttonHover,
                                                 .pressedColor = style.buttonActive,
                                                 .heldColor = style.buttonActive});
        if (item == 0) return false;

        GUITextRun* run = nullptr;
        setText(item, std::string(displayPart(label)), style.text);
        if ((run = getTextMutable(item)) != nullptr) {
            // left-aligned with a small inset, vertically centred — a row, not a button
            run->padding = glm::vec2(4.0f, std::round(style.framePaddingY * 0.5f));
        }
        setLastItem(item, id);
        return clickedElement == item;
    }

    /*
    Collapsing header. Returns whether the body should be drawn, and remembers its own open state
    across frames in the item slot — the caller keeps no bool of its own, exactly as with ImGui.
    */
    bool collapsingHeader(std::string_view label, bool defaultOpen = false) {
        uint64_t id = hashID(label, currentID());
        uint32_t window = currentWindow();
        GUILayout* layout = getLayout(window);
        GUIRect* c = get(window);
        if (layout == nullptr || c == nullptr) return false;

        GUIItemSlot& slot = imItems[id];
        if (!slot.openInitialised) {
            slot.open = defaultOpen;
            slot.openInitialised = true;
        }

        float width = std::max(1.0f, c->size.x - layout->style.padding.x * 2.0f - layout->indent);
        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(width, getLineHeight() + style.framePaddingY * 2.0f),
                                                 .color = style.button,
                                                 .hoverColor = style.buttonHover,
                                                 .pressedColor = style.buttonActive,
                                                 .heldColor = style.buttonActive});
        if (item == 0) return false;

        // re-read: acquireItem may have reallocated the slot table out from under `slot`
        if (clickedElement == item) imItems[id].open = !imItems[id].open;
        bool open = imItems[id].open;

        // ASCII marker — the atlas has no triangle glyph, and this is what ImGui's own bitmap font
        // fell back to before it packed its own
        setText(item, std::string(open ? "v " : "> ") + std::string(displayPart(label)), style.text);
        if (GUITextRun* run = getTextMutable(item)) run->padding = glm::vec2(6.0f, style.framePaddingY);

        setLastItem(item, id);
        return open;
    }

    /*
    Tree node. Returns true when open, in which case the caller must treePop() — the same
    asymmetric contract ImGui has, kept deliberately so the recursive node-tree call site ports
    without restructuring. A Leaf never opens and must not be popped.
    */
    bool treeNodeEx(std::string_view label, GUITreeFlags flags = GUITreeFlags::TreeNone) {
        uint64_t id = hashID(label, currentID());
        uint32_t window = currentWindow();
        GUILayout* layout = getLayout(window);
        GUIRect* c = get(window);
        if (layout == nullptr || c == nullptr) return false;

        bool leaf = (flags & GUITreeFlags::TreeLeaf) != 0;
        GUIItemSlot& slot = imItems[id];
        if (!slot.openInitialised) {
            slot.open = (flags & GUITreeFlags::TreeDefaultOpen) != 0;
            slot.openInitialised = true;
        }

        float width = std::max(1.0f, c->size.x - layout->style.padding.x * 2.0f - layout->indent);
        glm::vec4 base = (flags & GUITreeFlags::TreeSelected) ? style.buttonActive : glm::vec4(0);
        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(width, getLineHeight() + style.framePaddingY),
                                                 .color = base,
                                                 .hoverColor = style.buttonHover,
                                                 .pressedColor = style.buttonActive,
                                                 .heldColor = style.buttonActive});
        if (item == 0) return false;

        if (clickedElement == item && !leaf) imItems[id].open = !imItems[id].open;
        bool open = !leaf && imItems[id].open;

        std::string marker = leaf ? "  " : (open ? "v " : "> ");
        setText(item, marker + std::string(displayPart(label)), style.text);
        if (GUITextRun* run = getTextMutable(item)) run->padding = glm::vec2(4.0f, std::round(style.framePaddingY * 0.5f));

        setLastItem(item, id);
        if (open) {
            indent(window);
            pushID(label);
        }
        return open;
    }

    bool treeNode(std::string_view label) { return treeNodeEx(label, GUITreeFlags::TreeNone); }

    void treePop() {
        unindent(currentWindow());
        popID();
    }

    /*
    Tooltip for the widget just written, shown once the cursor has rested on it.

    Deferred to endFrame rather than built here: a tooltip has to draw over everything, and the
    only way to be last in a tree whose draw order is sibling order is to be added last.
    */
    void setItemTooltip(std::string_view text) {
        if (lastItem == 0 || hoveredElement != lastItem || text.empty()) return;
        std::chrono::duration<float> rested = std::chrono::steady_clock::now() - hoverSince;
        if (rested.count() < style.tooltipDelay) return;
        pendingTooltip.assign(text);
    }

    /*
    Dropdown. Returns true while open, in which case the caller lists its items and calls
    endCombo — the ImGui shape, so those call sites port unchanged.

    The popup hangs off element 0 rather than off the combo, for two reasons: it has to escape the
    parent window's clip box, and it has to outrank every window in draw order. Re-appending it to
    element 0 each frame is what keeps it on top.
    */
    bool beginCombo(std::string_view label, std::string_view preview) {
        uint64_t id = hashID(label, currentID());
        GUILayout* layout = getLayout(currentWindow());
        float width = (layout != nullptr && layout->nextItemWidth > 0.0f) ? layout->nextItemWidth : style.itemWidth;
        float height = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button,
                                                 .size = glm::vec2(width, height),
                                                 .color = style.frameBg,
                                                 .hoverColor = style.frameBgHover,
                                                 .pressedColor = style.buttonActive,
                                                 .heldColor = style.buttonActive});
        if (item == 0) return false;

        if (clickedElement == item) imItems[id].open = !imItems[id].open;
        setText(item, std::string(preview) + "  v", style.text);
        if (GUITextRun* run = getTextMutable(item)) run->padding = glm::vec2(6.0f, style.framePaddingY);

        labelAfter(label);
        setLastItem(item, id);
        if (!imItems[id].open) return false;

        GUIRect* button = get(item);
        if (button == nullptr) return false;

        // directly under the combo, in absolute screen px — element 0's top-left is the origin
        uint32_t popup = acquireOverlay(hashID("##popup", id),
                                        GUIRect{.size = glm::vec2(width, style.popupMaxHeight),
                                                .offset = glm::vec2(button->left(), button->bottom() + 2.0f),
                                                .color = style.popupBg,
                                                .hoverColor = style.popupBg,
                                                .pressedColor = style.popupBg,
                                                .heldColor = style.popupBg,
                                                .anchor = GUIAnchor::Top | GUIAnchor::Left});
        if (popup == 0) return false;
        if (getLayout(popup) == nullptr) {
            beginLayout(popup, GUILayoutStyle{.padding = glm::vec2(4.0f, 4.0f)});
            setScrollable(popup, true);
        }
        resetLayout(popup);

        comboStack.push_back({id, popup});
        pushWindowScope(popup, id);
        return true;
    }

    void endCombo() {
        if (comboStack.empty()) return;
        auto [id, popup] = comboStack.back();
        comboStack.pop_back();

        // Size to the items, but never past the cap — beyond that it scrolls, which is why the
        // popup opted into scrolling when it was created.
        GUILayout* layout = getLayout(popup);
        if (layout != nullptr) {
            GUIRect* p = get(popup);
            if (p != nullptr) {
                p->size.y = std::min(layout->contentExtent.y + layout->style.padding.y * 2.0f, style.popupMaxHeight);
                p->size.x = std::max(p->size.x, layout->contentExtent.x + layout->style.padding.x * 2.0f);
            }
        }
        popWindowScope();

        // Click-outside-to-dismiss. The press edge, not the click, so dragging out of the popup
        // and releasing elsewhere doesn't count as picking something.
        if (pressStarted && captured != itemOf(id) && !isAncestorOf(popup, captured)) imItems[id].open = false;
    }

    // One row inside an open combo. True on the frame it is picked, and picking closes the popup.
    bool comboItem(std::string_view label, bool selected = false) {
        bool picked = selectable(label, selected);
        if (picked && !comboStack.empty()) imItems[comboStack.back().first].open = false;
        return picked;
    }

    // The whole dropdown in one call, over a fixed item list.
    bool combo(std::string_view label, int* current, const char* const items[], int count) {
        if (current == nullptr || count <= 0) return false;
        int selected = glm::clamp(*current, 0, count - 1);
        bool changed = false;
        if (beginCombo(label, items[selected])) {
            for (int i = 0; i < count; i++) {
                if (comboItem(items[i], i == selected) && i != *current) {
                    *current = i;
                    changed = true;
                }
            }
            endCombo();
        }
        return changed;
    }

    // Single-line text field. Writes into the caller's buffer in place.
    bool inputText(std::string_view label, char* buffer, size_t bufferSize) { return inputTextImpl(label, {}, buffer, bufferSize); }

    // Same, with placeholder text shown while the buffer is empty.
    bool inputTextWithHint(std::string_view label, std::string_view hint, char* buffer, size_t bufferSize) {
        return inputTextImpl(label, hint, buffer, bufferSize);
    }

    // Typed numeric field
    bool inputFloat(std::string_view label, float* value, const char* format = "%.3f") {
        if (value == nullptr) return false;
        uint64_t id = hashID(label, currentID());
        std::string& buffer = inputBuffers[id];
        if (buffer.size() < 64) buffer.resize(64, '\0');

        uint32_t existing = itemOf(id);
        if (existing == 0 || keyboardFocus != existing) snprintf(buffer.data(), buffer.size(), format, *value);

        if (!inputTextImpl(label, {}, buffer.data(), buffer.size())) return false;

        // Accept only a parse that consumed something: a half-typed "-" or "1.5e" leaves the
        // bound value alone rather than snapping it to zero mid-keystroke.
        char* parseEnd = nullptr;
        float parsed = strtof(buffer.data(), &parseEnd);
        if (parseEnd == buffer.data()) return false;
        *value = parsed;
        return true;
    }

    // RGBA editor: a live swatch plus one slider per channel.
    bool colorEdit4(std::string_view label, float* color) {
        if (color == nullptr) return false;
        pushID(label);
        bool changed = colorSwatchAndSliders(color, 4);
        popID();
        labelAfter(label);
        return changed;
    }

    bool colorPicker3(std::string_view label, float* color) {
        if (color == nullptr) return false;
        pushID(label);
        bool changed = colorSwatchAndSliders(color, 3);
        popID();
        labelAfter(label);
        return changed;
    }

    // ----- item state queries -----
    // All refer to the widget most recently called

    bool isItemHovered() const {
        const GUIRect* e = getConst(lastItem);
        return e != nullptr && e->hovered;
    }

    bool isItemClicked() const { return lastItem != 0 && clickedElement == lastItem; }

    // Owns the live press — true for every frame of a drag.
    bool isItemActive() const { return lastItem != 0 && captured == lastItem; }

    // The frame the press landed on it.
    bool isItemActivated() const { return lastItem != 0 && captured == lastItem && capturedPrev != lastItem; }

    /*
    The frame a drag ended on this item, and the value moved at some point during it.

    Load-bearing, not cosmetic: the particle emitter re-reserves its GPU pool on exactly this edge.
    Firing every frame instead would thrash the allocator; never firing would leave the pool the
    wrong size. Consumed by the read, so it reports once.
    */
    bool isItemDeactivatedAfterEdit() {
        // A typed value never went through a drag, but it is the same edge to the caller: the
        // widget settled on a new value this frame. Frame-scoped, so it reports for this frame only.
        if (lastItemID != 0 && justCommitted(lastItemID)) return true;
        if (lastItem == 0 || captured == lastItem || capturedPrev != lastItem) return false;
        auto it = imItems.find(lastItemID);
        if (it == imItems.end() || !it->second.editedWhileActive) return false;
        it->second.editedWhileActive = false;
        return true;
    }

    GUIStyle& getStyle() { return style; }

    // Centres a run inside its element's rect by padding it. A stopgap for real alignment, which
    // belongs to the layout engine — but a label sitting in the corner of its button is unreadable,
    // and every button has one.
    void centerText(uint32_t index) {
        GUIRect* e = get(index);
        GUITextRun* run = getTextMutable(index);
        if (e == nullptr || run == nullptr) return;
        run->padding = glm::max((e->size - font.measure(run->text)) * 0.5f, glm::vec2(0));
    }

    // px bounding box of a run, honouring newlines and tabs exactly as the renderer does. Layout
    // cannot size anything to its label without this.
    glm::vec2 measureText(std::string_view text) const { return font.measure(text); }
    float getLineHeight() const { return font.lineHeight(); }
    const GUIFont& getFont() const { return font; }

    /*
    Behaviour for the types GUI ships with. Called from init(), before anything is added, because
    onAdd doesn't fire retroactively.
    */
    void registerBuiltinHandlers() {
        // Bridges the type-keyed registry to the per-element table. No internalType, so this stays
        // off the resize grips — they're Buttons too, and a grip firing a user action would be a
        // surprise.
        registerHandler({.type = GUIType::Button, .onInput = [](GUI& g, uint32_t index) {
            if (g.getClicked() == index) g.runAction(index);
        }});

        // A Resizeable grows its own four corner grips.
        registerHandler({.type = GUIType::Resizeable, .onAdd = [](GUI& g, uint32_t panel) {
            const GUIAnchor corners[4] = {GUIAnchor::Top | GUIAnchor::Left, GUIAnchor::Top | GUIAnchor::Right,
                                          GUIAnchor::Bottom | GUIAnchor::Left, GUIAnchor::Bottom | GUIAnchor::Right};
            for (GUIAnchor corner : corners) {
                g.addElement(GUIRect{.type = GUIType::Button | GUIType::Internal,
                                     .internalType = InternalType::ResizeableHandle,
                                     .size = glm::vec2(GUI_RESIZE_GRIP_PX),
                                     .color = glm::vec4(0),
                                     .hoverColor = glm::vec4(1, 1, 1, 0.25f),
                                     .pressedColor = glm::vec4(1, 1, 1, 0.45f),
                                     .heldColor = glm::vec4(1, 1, 1, 0.45f),
                                     .anchor = corner},
                             panel);
            }
        }});

        // The grip drives its parent. The subtype in the key is what keeps this off a panel's
        // ordinary buttons — grips are Buttons too, so `type` alone wouldn't separate them.
        registerHandler({.type = GUIType::Internal, .internalType = InternalType::ResizeableHandle,
                         .onInput = [](GUI& g, uint32_t index) {
            GUIRect* grip = g.get(index);
            if (grip == nullptr || !grip->held) return;

            GUIRect* panel = g.get(grip->parent);
            if (panel == nullptr) return;

            glm::vec2 mouse = InputManager::getCurrentState().mouseDelta;
            if (mouse == glm::vec2(0)) return;

            // which edges this corner owns: -1 = the low side, +1 = the high side
            glm::vec2 edge(grip->anchor & GUIAnchor::Left ? -1.0f : 1.0f,
                           grip->anchor & GUIAnchor::Top ? -1.0f : 1.0f);

            // Clamp first, then re-derive the delta that actually landed. Otherwise travel spent
            // pushing against the floor has to be un-travelled before the panel grows again.
            glm::vec2 newSize = glm::max(panel->size + edge * mouse, glm::vec2(GUI_MIN_RESIZE_PX));
            glm::vec2 applied = edge * (newSize - panel->size);

            // Dragging one corner has to pin the opposite one, which means center moves half the
            // delta whichever corner it is. offset then absorbs whatever share of that the panel's
            // own anchor doesn't already account for: none of it when the anchor is centered on that
            // axis, all of it when the anchor sits on the very edge being dragged.
            glm::vec2 pull(panel->anchor & GUIAnchor::Left    ?  1.0f
                           : panel->anchor & GUIAnchor::Right ? -1.0f : 0.0f,
                           panel->anchor & GUIAnchor::Top      ?  1.0f
                           : panel->anchor & GUIAnchor::Bottom ? -1.0f : 0.0f);

            panel->offset += applied * 0.5f * (glm::vec2(1.0f) - pull * edge);
            panel->size = newSize;
        }});
    }

    // grows never shrinks, recycles unused slots. parent 0 anchors against the screen.
    // returns 0 (the dead sentinel) if full or the parent is bad.
    uint32_t addElement(const GUIRect& element, uint32_t parent = 0) {
        if (parent >= elements.size()) return 0;
        if (parent != 0 && !elements[parent].alive()) return 0;

        uint32_t index;
        if (!freeSlots.empty()) {
            index = freeSlots.front();
            freeSlots.pop();
            elements[index] = element;
        } else {
            if (elements.size() >= MAX_GUI_ELEMENTS) return 0;
            index = static_cast<uint32_t>(elements.size());
            elements.push_back(element);
        }
        GUIRect& e = elements[index];
        e.self = index;
        e.parent = parent;
        e.firstChild = 0;
        e.lastChild = 0;
        e.nextSibling = 0;
        linkChild(parent, index);

        // last, so a handler that spawns children sees a fully linked parent. It may reallocate
        // `elements`, which is why `e` isn't touched past this point.
        dispatch(&GUITypeHandler::onAdd, index);

        // Content added to a panel would otherwise sit past its grips in sibling order, which is
        // both draw order and (since the hit test takes the last match) hit order. A grip's own
        // insertion is exempt or this would recurse through the four it just spawned.
        if (!(element.internalType & InternalType::ResizeableHandle)) raiseResizeHandles(parent);

        return index;
    }

    // frees the element and its whole subtree
    void removeElement(uint32_t index) {
        if (index == 0 || index >= elements.size() || !elements[index].alive()) return;
        unlinkChild(elements[index].parent, index);

        scratch.clear();
        scratch.push_back(index);
        while (!scratch.empty()) {
            uint32_t n = scratch.back();
            scratch.pop_back();
            for (uint32_t c = elements[n].firstChild; c != 0; c = elements[c].nextSibling) {
                scratch.push_back(c);
            }
            // before markFree, so the handler can still read the rect it's cleaning up after
            dispatch(&GUITypeHandler::onRemove, n);
            elements[n].markFree();
            elements[n].parent = 0;
            elements[n].firstChild = 0;
            elements[n].lastChild = 0;
            elements[n].nextSibling = 0;
            // else a recycled slot would inherit the dead rect's press, its caret, or its action
            if (captured == n) captured = 0;
            if (clickedElement == n) clickedElement = 0;
            if (hoveredElement == n) hoveredElement = 0;
            if (keyboardFocus == n) {
                keyboardFocus = 0;
                guiWantsKeyboard = false;
            }
            actions.erase(n);
            texts.erase(n);
            layouts.erase(n);
            freeSlots.push(n);
        }
    }

    // moves an element (and its subtree) under a new parent; appends, so it lands on top
    void setParent(uint32_t index, uint32_t newParent) {
        if (index == 0 || index >= elements.size() || !elements[index].alive()) return;
        if (newParent >= elements.size()) return;
        if (newParent != 0 && !elements[newParent].alive()) return;
        if (isAncestorOf(index, newParent)) return; // would build a cycle

        unlinkChild(elements[index].parent, index);
        elements[index].parent = newParent;
        elements[index].nextSibling = 0;
        linkChild(newParent, index);
    }

    GUIRect* get(uint32_t index) {
        if (index == 0 || index >= elements.size() || !elements[index].alive()) return nullptr;
        return &elements[index];
    }

    // Iterate children of an element via callback: fn(GUIRect& child). child.self is the handle.
    // The next sibling is cached before the callback, so fn may removeElement(child.self) —
    // but not a *different* sibling, which would cut the chain mid-walk.
    template <typename Fn>
    void forEachChild(uint32_t index, Fn&& fn) {
        if (index >= elements.size()) return;
        uint32_t c = elements[index].firstChild;
        while (c != 0) {
            uint32_t next = elements[c].nextSibling;
            fn(elements[c]);
            c = next;
        }
    }

    // Screen rect, as resolved by the last uploadGPU. Roots anchor against this.
    const GUIRect& getScreenRect() const { return elements[0]; }

    
    // Resolves every element's absolute `center` and clip box against its parent chain, top-down.
    void resolveLayout(glm::uvec2 resolution) {
        #if TRACY_ENABLE
        ZoneScopedN("gui resolveLayout"); // nests under uploadGPU when called from there
        #endif
        // element 0 is the screen: roots anchor against it, clip to it, and it never draws (self == 0)
        elements[0].size = glm::vec2(resolution);
        elements[0].center = elements[0].size * 0.5f;
        elements[0].clipMin = glm::vec2(0);
        elements[0].clipMax = elements[0].size;

        scratch.clear();
        scratch.push_back(0);
        // a well-formed tree pops each element exactly once; the cap just stops a malformed
        // one (cycle from a bad edit) from hanging the frame
        size_t popBudget = elements.size();
        while (!scratch.empty() && popBudget-- > 0) {
            uint32_t n = scratch.back();
            scratch.pop_back();
            const GUIRect& e = elements[n];

            // Children are trimmed to this rect on top of whatever already trimmed it, so nesting
            // accumulates down the chain rather than each level starting fresh — unless the rect
            // opts out, in which case it passes its own clip straight through (see clipsChildren).
            glm::vec2 childClipMin = e.clipMin;
            glm::vec2 childClipMax = e.clipMax;
            if (e.clipsChildren) {
                childClipMin = glm::max(childClipMin, glm::vec2(e.left(), e.top()));
                childClipMax = glm::min(childClipMax, glm::vec2(e.right(), e.bottom()));
            }

            // any DFS order works here — all that matters is a parent resolving before its
            // children, which the stack guarantees
            for (uint32_t c = elements[n].firstChild; c != 0; c = elements[c].nextSibling) {
                elements[c].center = resolveCenter(elements[c], elements[n]);
                elements[c].clipMin = childClipMin;
                elements[c].clipMax = childClipMax;
                scratch.push_back(c);
            }
        }
    }

    // Flattens the tree into this frame's slice of the quad buffer.
    void uploadGPU(BindlessSystem& bindless, uint32_t frameIndex, glm::uvec2 resolution) {
        #if TRACY_ENABLE
        ZoneScopedN("gui uploadGPU");
        #endif
        quadStaging.clear();
        resolveLayout(resolution);

        scratch.clear();
        scratch.push_back(0);
        size_t popBudget = elements.size();
        while (!scratch.empty() && popBudget-- > 0) {
            uint32_t n = scratch.back();
            scratch.pop_back();
            const GUIRect& e = elements[n];

            // a rect trimmed to nothing costs no instance — the clip doubles as the cull
            GPUGuiQuad quad{}; // zeroes the pad words the old designated init used to
            if (e.alive() && quadStaging.size() < MAX_GUI_QUADS &&
                clipQuad(glm::vec2(e.left(), e.top()), glm::vec2(e.right(), e.bottom()), e.uvMin, e.uvMax, e.clipMin, e.clipMax, quad)) {
                quad.color = e.held ? e.heldColor : e.pressed ? e.pressedColor : e.hovered ? e.hoverColor : e.color;
                quad.textureIndex = e.textureIndex == GUI_ATLAS_TEXTURE ? atlasTextureIndex : e.textureIndex;
                quad.samplerIndex = e.samplerIndex == GUI_DEFAULT_SAMPLER ? nearestSamplerIndex : e.samplerIndex;
                quadStaging.push_back(quad);
            }

            // after the element's own quad so the label sits on its button, before its children so
            // a child element still draws over it
            if (e.alive() && !texts.empty()) {
                auto run = texts.find(n);
                if (run != texts.end()) emitTextQuads(e, run->second);
            }

            // push children reversed so they pop back in insertion order
            size_t firstPushed = scratch.size();
            for (uint32_t c = elements[n].firstChild; c != 0; c = elements[c].nextSibling) {
                scratch.push_back(c);
            }
            std::reverse(scratch.begin() + firstPushed, scratch.end());
        }

        quadCount = static_cast<uint32_t>(quadStaging.size());
        if (quadCount == 0) return;
        bindless.descriptorSet->writeFixedBuffer<GPUGuiQuad>(GPUBufferIndex, quadStaging.data(), quadCount, frameIndex * MAX_GUI_QUADS, frameIndex);
    }

    // Records into an already-open rendering scope (see Renderer::recordOverlayPass).
    void record(BindlessSystem& bindless, vk::raii::CommandBuffer& cmd, uint32_t frameIndex, glm::uvec2 resolution) {
        #if TRACY_ENABLE
        ZoneScopedN("gui record"); // CPU-side recording cost only; the GPU pass has its own zone
        #endif
        if (quadCount == 0 || pipelineIndex == 0xFFFFFFFF) return;

        auto& pipeline = bindless.pipelineManager->getPostProcessPipelines()[pipelineIndex];
        bindPipeline(cmd, *pipeline);

        GUIPushConstants pc = {.quadBufferAddress = bindless.descriptorSet->getFixedBuffers()[GPUBufferIndex]->address +
                                                    static_cast<vk::DeviceSize>(frameIndex) * MAX_GUI_QUADS * sizeof(GPUGuiQuad),
                               .resolution = resolution,
                               .quadCount = quadCount};
        cmd.pushConstants<GUIPushConstants>(*pipeline->layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, pc);
        cmd.draw(6, quadCount, 0, 0);
    }

    /*
    Resolves what the cursor is over and who owns the live press, and publishes the two capture
    flags. Call after resolveLayout and BEFORE InputManager::tickInputState so that the scene 
    raycast gets to see wantsMouse() before it consumes the click.
    */
    void hitTest() {
        #if TRACY_ENABLE
        ZoneScopedN("gui hitTest");
        #endif
        const auto& in = InputManager::getCurrentState();
        const auto& prevIn = InputManager::getPreviousState();
        glm::vec2 pos = in.mousePos;
        // the input manager latches the last mouse *event*, so these stay true for the whole hold
        bool down = in.mouse_button == GLFW_MOUSE_BUTTON_LEFT && in.mouse_action == GLFW_PRESS;
        bool prevDown = prevIn.mouse_button == GLFW_MOUSE_BUTTON_LEFT && prevIn.mouse_action == GLFW_PRESS;

        // against the *clipped* rect — a child trimmed away by its parent isn't on screen, so it
        // shouldn't answer the cursor either
        auto inside = [pos](const GUIRect& r) {
            return pos.x > glm::max(r.left(), r.clipMin.x) && pos.x < glm::min(r.right(), r.clipMax.x) &&
                   pos.y > glm::max(r.top(), r.clipMin.y) && pos.y < glm::min(r.bottom(), r.clipMax.y);
        };

        uint32_t hit = 0;
        scratch.clear();
        scratch.push_back(0);
        size_t popBudget = elements.size();
        while (!scratch.empty() && popBudget-- > 0) {
            uint32_t n = scratch.back();
            scratch.pop_back();
            if (n != 0 && elements[n].alive() && elements[n].hitTestable && inside(elements[n])) hit = n;

            size_t firstPushed = scratch.size();
            for (uint32_t c = elements[n].firstChild; c != 0; c = elements[c].nextSibling) {
                scratch.push_back(c);
            }
            std::reverse(scratch.begin() + firstPushed, scratch.end());
        }

        // before anything moves it: isItemActivated / isItemDeactivatedAfterEdit are edge queries,
        // and the edge is only visible by comparing against last frame's owner
        capturedPrev = captured;

        if (down && !prevDown) {
            captured = hit; // the press edge picks the target, nothing else can
            // Clicking anywhere else drops the caret, including on empty space — same as every
            // other editor. Only a TextBox can take it.
            keyboardFocus = (hit != 0 && (elements[hit].type & GUIType::TextBox)) ? hit : 0;
        }
        // a click is a release back inside the rect the press started on
        clickedElement = (!down && prevDown && captured != 0 && captured == hit) ? captured : 0;
        if (!down) captured = 0;

        // Second click on the same rect, close enough behind the first. The pair is cleared once it
        // fires, so a third click starts counting again instead of reporting a second double click.
        doubleClickedElement = 0;
        if (clickedElement != 0) {
            auto now = std::chrono::steady_clock::now();
            if (clickedElement == lastClickElement &&
                std::chrono::duration<float>(now - lastClickTime).count() < style.doubleClickTime) {
                doubleClickedElement = clickedElement;
                lastClickElement = 0;
            } else {
                lastClickElement = clickedElement;
            }
            lastClickTime = now;
        }

        // Flat walk: reaches every depth, and a dead slot gets its flags cleared so a recycled
        // element can't inherit a stale hover.
        for (uint32_t i = 1; i < elements.size(); i++) {
            GUIRect& r = elements[i];
            if (!r.alive()) {
                r.hovered = r.pressed = r.held = false;
                continue;
            }
            bool over = i == hit;
            bool active = i == captured;
            // while a press is live, nothing but its own target reacts to the cursor
            r.hovered = over && (captured == 0 || active);
            // drag off the captured rect un-presses it; drag back re-presses. never leaks to a neighbour.
            r.pressed = active && over;
            // owns the live press, cursor position irrelevant — this is the drag flag. A drag
            // outruns whatever it's moving, so gating this on `over` would drop the drag the moment
            // the cursor got ahead of the rect.
            r.held = active && prevDown;
        }

        // Tooltips need "how long has the cursor rested here", so the clock restarts whenever the
        // hovered element changes rather than counting from any absolute point.
        if (hit != hoveredElement) hoverSince = std::chrono::steady_clock::now();
        // popups close on a press that lands outside them, which needs the press *edge*
        pressStarted = down && !prevDown;
        mouseDown = down;
        hoveredElement = hit;
        // A live press keeps the claim even once the cursor leaves the rect, so dragging a resize
        // grip out over the viewport can't start spinning the camera halfway through.
        guiWantsMouse = hit != 0 || captured != 0;
        guiWantsKeyboard = keyboardFocus != 0;
    }

    /*
    Runs the registered onInput handlers once every flag in the tree is settled
    Flat walk of the tree.
    */
    void runBehaviour() {
        #if TRACY_ENABLE
        ZoneScopedN("gui runBehaviour");
        #endif
        // first: everything below should act on the order the click just established
        raiseFocusedWindow();
        // before the handlers: a handler that resizes a panel should see this frame's scroll
        // already clamped against the size it is about to change
        updateScroll();

        for (uint32_t i = 1; i < elements.size(); i++) {
            if (!elements[i].alive() || elements[i].type == GUIType::Default) continue;
            dispatch(&GUITypeHandler::onInput, i);
        }
    }

    // Element clicked this frame (press and release both inside it), 0 if none. 
    // Valid until the next hitTest.
    uint32_t getClicked() const { return clickedElement; }

    // Topmost element under the cursor, 0 if none. 
    // Valid until the next hitTest.
    uint32_t getHovered() const { return hoveredElement; }

    // Whether the GUI has claimed this frame's input.    
    bool wantsMouse() const { return guiWantsMouse; }
    bool wantsKeyboard() const { return guiWantsKeyboard; }

    // Left button held, as hitTest resolved it
    bool isMouseDown() const { return mouseDown; }

    // Element holding the text caret, 0 if none. Set by clicking a TextBox, cleared by clicking
    // anything else.
    uint32_t getKeyboardFocus() const { return keyboardFocus; }

    uint32_t getAtlasTextureIndex() const { return atlasTextureIndex; }

    // instances the last uploadGPU produced — rects plus every glyph they expanded to
    uint32_t getQuadCount() const { return quadCount; }

    // Assign to GUIRect::samplerIndex to override the per-rect default (Nearest/clamp).
    uint32_t getNearestSampler() const { return nearestSamplerIndex; }
    uint32_t getLinearSampler() const { return linearSamplerIndex; }

private:

    // No-op when nothing is bound
    void runAction(uint32_t index) {
        auto it = actions.find(index);
        if (it == actions.end()) return;
        GUIHook action = it->second;
        action(*this, index);
    }

    // Runs one hook across every handler the element matches on both axes. `hook` is a pointer-to-member
    // (&GUITypeHandler::onAdd, ...). The keys are copied up front because a handler is allowed to add
    // elements, which can reallocate the vector out from under a reference.
    void dispatch(GUIHook GUITypeHandler::*hook, uint32_t index) {
        if (index == 0 || index >= elements.size()) return;
        GUIType type = elements[index].type;
        InternalType internalType = elements[index].internalType;
        for (GUITypeHandler& h : handlers) {
            if ((type & h.type) && (internalType & h.internalType) && h.*hook) (h.*hook)(*this, index);
        }
    }

    // Fills in a quad's geometry trimmed to a clip box, or returns false if nothing survives the trim.    
    static bool clipQuad(glm::vec2 rectMin, glm::vec2 rectMax, glm::vec2 uvMin, glm::vec2 uvMax,
                         glm::vec2 clipMin, glm::vec2 clipMax, GPUGuiQuad& out) {
        out.minPx = glm::max(rectMin, clipMin);
        out.maxPx = glm::min(rectMax, clipMax);
        // also catches a degenerate rect, so the division below can't hit a zero span
        if (out.minPx.x >= out.maxPx.x || out.minPx.y >= out.maxPx.y) return false;

        glm::vec2 span = rectMax - rectMin;
        glm::vec2 uvSpan = uvMax - uvMin;
        out.uvMin = uvMin + uvSpan * ((out.minPx - rectMin) / span);
        out.uvMax = uvMin + uvSpan * ((out.maxPx - rectMin) / span);
        return true;
    }

    // Expands one text run into glyph quads, appended straight to this frame's instance list.    
    void emitTextQuads(const GUIRect& e, const GUITextRun& run) {
        if (!font.loaded() || run.text.empty()) return;

        float startX = std::round(e.left() + run.padding.x);
        // pen sits on the baseline; the run's box starts at the element's top-left
        glm::vec2 pen(startX, std::round(e.top() + run.padding.y + font.ascent()));

        for (char raw : run.text) {
            if (quadStaging.size() >= MAX_GUI_QUADS) return;
            unsigned char c = static_cast<unsigned char>(raw);

            if (c == '\n') {
                pen.x = startX;
                pen.y += font.lineHeight();
                continue;
            }
            if (c == '\t') {
                pen.x += font.spaceAdvance() * GUI_TAB_SPACES;
                continue;
            }

            const GUIGlyph* g = font.glyph(c);
            if (g == nullptr) continue;

            // whitespace has no bitmap but still moves the pen
            if (g->size.x > 0.0f && g->size.y > 0.0f) {
                // rounded per glyph, while the pen itself stays fractional — snapping the pen
                // instead would accumulate the rounding error into visibly uneven spacing
                glm::vec2 glyphMin = glm::round(pen + g->offset);
                GPUGuiQuad quad{};
                if (clipQuad(glyphMin, glyphMin + g->size, g->uvMin, g->uvMax, e.clipMin, e.clipMax, quad)) {
                    quad.color = run.color;
                    quad.textureIndex = atlasTextureIndex;
                    quad.samplerIndex = nearestSamplerIndex;
                    quadStaging.push_back(quad);
                }
            }
            pen.x += g->advance;
        }
    }

    // anchor picks a point on the parent and the child's matching corner; offset displaces from there
    static glm::vec2 resolveCenter(const GUIRect& e, const GUIRect& parent) {
        glm::vec2 anchorPoint;
        glm::vec2 toCenter; // child's aligned corner -> its center

        if (e.anchor & GUIAnchor::Left)        { anchorPoint.x = parent.left();    toCenter.x =  e.size.x / 2; }
        else if (e.anchor & GUIAnchor::Right)  { anchorPoint.x = parent.right();   toCenter.x = -e.size.x / 2; }
        else                                   { anchorPoint.x = parent.center.x;  toCenter.x =  0; }

        if (e.anchor & GUIAnchor::Top)         { anchorPoint.y = parent.top();     toCenter.y =  e.size.y / 2; }
        else if (e.anchor & GUIAnchor::Bottom) { anchorPoint.y = parent.bottom();  toCenter.y = -e.size.y / 2; }
        else                                   { anchorPoint.y = parent.center.y;  toCenter.y =  0; }

        return anchorPoint + toCenter + e.offset;
    }

    // append, so sibling order stays insertion order (and therefore draw order)
    void linkChild(uint32_t parent, uint32_t child) {
        if (elements[parent].firstChild == 0) {
            elements[parent].firstChild = child;
        } else {
            elements[elements[parent].lastChild].nextSibling = child;
        }
        elements[parent].lastChild = child;
    }

    void unlinkChild(uint32_t parent, uint32_t child) {
        if (parent >= elements.size()) return;
        if (elements[parent].firstChild == child) {
            elements[parent].firstChild = elements[child].nextSibling;
            if (elements[parent].lastChild == child) elements[parent].lastChild = 0;
            return;
        }
        for (uint32_t c = elements[parent].firstChild; c != 0; c = elements[c].nextSibling) {
            if (elements[c].nextSibling == child) {
                elements[c].nextSibling = elements[child].nextSibling;
                if (elements[parent].lastChild == child) elements[parent].lastChild = c;
                return;
            }
        }
    }

    // ----- immediate-mode internals -----

    // FNV-1a, seeded with the enclosing scope's id so the same label nests without colliding.
    static uint64_t hashID(std::string_view label, uint64_t seed) {
        uint64_t h = seed != 0 ? seed : 1469598103934665603ull;
        for (char c : label) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ull;
        }
        return h;
    }

    // "Radius##3" displays "Radius" but hashes the whole thing, so two nodes sharing a name still get distinct ids.
    static std::string_view displayPart(std::string_view label) {
        size_t pos = label.find("##");
        return pos == std::string_view::npos ? label : label.substr(0, pos);
    }

    uint64_t currentID() const { return idStack.empty() ? 0 : idStack.back(); }

    // Every isItemXxx query answers about the widget the caller just wrote. A widget that emits a
    // trailing caption has to put this back, or the queries would answer about the caption.
    void setLastItem(uint32_t element, uint64_t id) {
        lastItem = element;
        lastItemID = id;
    }

    // The caption ImGui draws to the right of a framed widget. Scoped under the widget's own id so
    // two widgets sharing a caption don't collide.
    void labelAfter(std::string_view label) {
        std::string_view display = displayPart(label);
        if (display.empty()) return;
        sameLine(currentWindow());
        pushID(label);
        textColored(display, style.text);
        popID();
        // A framed widget is framePaddingY taller than its own text, so a caption placed at the
        // row's top edge sits high against it. Nudge it onto the frame's centre line.
        if (GUIRect* caption = get(lastItem)) caption->offset.y += style.framePaddingY;
    }

    void addLabel(std::string_view label) {
        std::string_view display = displayPart(label);
        if (display.empty()) return;
        pushID(label);
        textColored(display, style.text);
        popID();
        // A framed widget is framePaddingY taller than its own text, so a caption placed at the
        // row's top edge sits high against it. Nudge it onto the frame's centre line.
        if (GUIRect* caption = get(lastItem)) caption->offset.y += style.framePaddingY;
    }

    // Id for an item with no label of its own 
    uint64_t nextAutoID(std::string_view kind) {
        uint64_t id = hashID(kind, currentID());
        uint32_t n = autoID++;
        return hashID(std::string_view(reinterpret_cast<const char*>(&n), sizeof(n)), id);
    }

    uint32_t itemOf(uint64_t id) const {
        auto it = imItems.find(id);
        return it == imItems.end() ? 0 : it->second.element;
    }

    /*
    Centered caption on top of a widget that has decoration of its own.
    Widgets with no decoration (a button) can still use setText directly.
    */
    void captionChild(uint64_t id, uint32_t parent, glm::vec2 size, std::string_view text, glm::vec4 color) {
        uint32_t item = acquireChild(id, parent, GUIRect{.size = size,
                                                          .color = glm::vec4(0),
                                                          .hoverColor = glm::vec4(0),
                                                          .pressedColor = glm::vec4(0),
                                                          .heldColor = glm::vec4(0),
                                                          .hitTestable = false,
                                                          .anchor = GUIAnchor::Top | GUIAnchor::Left});
        if (item == 0) return;
        setText(item, std::string(text), color);
        centerText(item);
    }

    bool buttonImpl(std::string_view label, glm::vec2 size, glm::vec4 base, glm::vec4 hover) {
        std::string_view display = displayPart(label);
        glm::vec2 wanted = size;
        if (wanted.x < 0.0f) wanted.x = availableWidth();
        else if (wanted.x == 0.0f) wanted.x = measureText(display).x + 16.0f;
        if (wanted.y <= 0.0f) wanted.y = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t item = acquireItem(hashID(label, currentID()), GUIRect{.type = GUIType::Button,
                                                                        .size = wanted,
                                                                        .color = base,
                                                                        .hoverColor = hover,
                                                                        .pressedColor = style.buttonActive,
                                                                        .heldColor = style.buttonActive});
        if (item == 0) return false;
        setText(item, std::string(display), style.text);
        centerText(item);
        setLastItem(item, hashID(label, currentID()));
        return clickedElement == item;
    }

    // Which half of a two-part widget the item queries should answer about: the one owning or
    // just released from the press, else the first.
    uint64_t subItemInFocus(uint64_t first, uint64_t second) const {
        uint32_t e = itemOf(second);
        return (e != 0 && (captured == e || capturedPrev == e)) ? second : first;
    }

    const GUIRect* getConst(uint32_t index) const {
        if (index == 0 || index >= elements.size() || !elements[index].alive()) return nullptr;
        return &elements[index];
    }

    /*
    Overwrites an existing element with this frame's prototype while keeping what the element owns
    rather than the caller: its place in the tree, the input state hitTest just resolved, and the
    resolved layout.
    */
    static void applyPrototype(GUIRect& dst, const GUIRect& prototype) {
        uint32_t self = dst.self, parent = dst.parent, next = dst.nextSibling;
        uint32_t first = dst.firstChild, last = dst.lastChild;
        bool hovered = dst.hovered, pressed = dst.pressed, held = dst.held;
        glm::vec2 center = dst.center, clipMin = dst.clipMin, clipMax = dst.clipMax;

        dst = prototype;

        dst.self = self;
        dst.parent = parent;
        dst.nextSibling = next;
        dst.firstChild = first;
        dst.lastChild = last;
        dst.hovered = hovered;
        dst.pressed = pressed;
        dst.held = held;
        dst.center = center;
        dst.clipMin = clipMin;
        dst.clipMax = clipMax;
    }

    // Find-or-create the element behind one widget call, placed at the current window's cursor.
    uint32_t acquireItem(uint64_t id, const GUIRect& prototype) {
        uint32_t container = currentWindow();
        GUILayout* layout = getLayout(container);
        if (layout == nullptr) return 0;
        uint32_t root = ensureContentRoot(container, *layout);
        if (root == 0) return 0;
        // ensureContentRoot adds an element, and an onAdd handler is allowed to create layouts of
        // its own — which would rehash the map and dangle the pointer above
        layout = getLayout(container);
        if (layout == nullptr) return 0;

        GUIRect placed = prototype;
        if (layout->nextItemWidth > 0.0f) placed.size.x = layout->nextItemWidth;
        layout->nextItemWidth = 0.0f;
        placed.anchor = GUIAnchor::Top | GUIAnchor::Left;
        placed.offset = advanceCursor(*layout, placed.size);

        GUIItemSlot& slot = imItems[id];
        slot.lastFrame = frameCounter;
        if (get(slot.element) == nullptr) {
            slot.element = addElement(placed, root);
        } else {
            if (elements[slot.element].parent != root) setParent(slot.element, root);
            applyPrototype(elements[slot.element], placed);
        }

        lastItem = slot.element;
        lastItemID = id;
        return slot.element;
    }

    // Same, for a widget's internal parts (a slider's fill bar). Anchored inside its parent rather
    // than placed by the cursor, so it doesn't disturb the flow.
    uint32_t acquireChild(uint64_t id, uint32_t parent, const GUIRect& prototype) {
        if (get(parent) == nullptr) return 0;

        GUIItemSlot& slot = imItems[id];
        slot.lastFrame = frameCounter;
        if (get(slot.element) == nullptr) {
            slot.element = addElement(prototype, parent);
        } else {
            if (elements[slot.element].parent != parent) setParent(slot.element, parent);
            applyPrototype(elements[slot.element], prototype);
        }
        return slot.element;
    }

    // Scope shared by windows and popups: ids nest, and unlabelled items count from zero inside
    // each so a separator in a popup can't collide with one in the window behind it.
    void pushWindowScope(uint32_t element, uint64_t id) {
        idStack.push_back(id);
        windowStack.push_back(element);
        autoIDStack.push_back(autoID);
        autoID = 0;
    }

    void popWindowScope() {
        if (!idStack.empty()) idStack.pop_back();
        if (!windowStack.empty()) windowStack.pop_back();
        if (!autoIDStack.empty()) {
            autoID = autoIDStack.back();
            autoIDStack.pop_back();
        }
    }

    // Find-or-create an element hanging directly off the screen rect, re-appended every frame.
    uint32_t acquireOverlay(uint64_t id, const GUIRect& prototype) {
        GUIItemSlot& slot = imItems[id];
        slot.lastFrame = frameCounter;
        if (get(slot.element) == nullptr) {
            slot.element = addElement(prototype, 0);
        } else {
            applyPrototype(elements[slot.element], prototype);
            setParent(slot.element, 0);
        }
        return slot.element;
    }

    // Title bar, collapse toggle and close button. Returns whether the window is collapsed.
    bool drawWindowChrome(uint32_t panel, uint64_t id, const std::string& title, bool* open) {
        GUIRect* p = get(panel);
        if (p == nullptr) return false;

        float barHeight = titleBarHeight();
        float width = p->size.x;
        bool focused = focusedWindow == panel;
        glm::vec4 barColor = focused ? style.titleBarActive : style.titleBar;

        uint32_t bar = acquireChild(hashID("##titlebar", id), panel,
                                    GUIRect{.type = GUIType::Button,
                                            .size = glm::vec2(width, barHeight),
                                            .color = barColor,
                                            .hoverColor = barColor,
                                            .pressedColor = barColor,
                                            .heldColor = barColor,
                                            .anchor = GUIAnchor::Top | GUIAnchor::Left});
        if (bar == 0) return windowStates[id].collapsed;

        // Drag to move. offset always maps +x right and +y down whatever the anchor is (see
        // resolveCenter), so this needs none of the anchor compensation the resize grips do.
        if (!(windowStates[id].flags & GUIWindowNoMove)) {
            if (GUIRect* barRect = get(bar); barRect != nullptr && barRect->held) {
                if (GUIRect* panelRect = get(panel)) panelRect->offset += InputManager::getCurrentState().mouseDelta;
            }
        }

        bool wasCollapsed = windowStates[id].collapsed;
        bool collapsed = wasCollapsed;
        bool canCollapse = !(windowStates[id].flags & GUIWindowNoCollapse);

        setText(bar, title, style.titleText);
        if (GUITextRun* run = getTextMutable(bar)) {
            run->padding = glm::vec2(canCollapse ? barHeight + 2.0f : 8.0f, style.framePaddingY);
        }

        if (canCollapse) {
            uint32_t toggle = acquireChild(hashID("##collapse", id), bar,
                                           GUIRect{.type = GUIType::Button,
                                                   .size = glm::vec2(barHeight),
                                                   .color = glm::vec4(0),
                                                   .hoverColor = style.buttonHover,
                                                   .pressedColor = style.buttonActive,
                                                   .heldColor = style.buttonActive,
                                                   .anchor = GUIAnchor::Top | GUIAnchor::Left});
            if (clickedElement == toggle) collapsed = !collapsed;
            setText(toggle, collapsed ? ">" : "v", style.titleText);
            centerText(toggle);
        }

        if (open != nullptr) {
            uint32_t close = acquireChild(hashID("##close", id), bar,
                                          GUIRect{.type = GUIType::Button,
                                                  .size = glm::vec2(barHeight),
                                                  .color = glm::vec4(0),
                                                  .hoverColor = style.closeHover,
                                                  .pressedColor = style.closeHover,
                                                  .heldColor = style.closeHover,
                                                  .anchor = GUIAnchor::Top | GUIAnchor::Right});
            if (clickedElement == close) *open = false;
            setText(close, "x", style.titleText);
            centerText(close);
        }

        // capture height before collapsing
        if (GUIRect* panelRect = get(panel)) {
            if (!wasCollapsed) windowStates[id].restoreHeight = std::max(panelRect->size.y, barHeight);
            panelRect->size.y = collapsed ? barHeight : std::max(windowStates[id].restoreHeight, barHeight);
        }
        return collapsed;
    }

    /*
    Click-to-front: a press anywhere in a window moves that window to the tail of element 0's children
    Runs before the frame's widget calls, so the raise is visible in the same frame it is asked for.
    Popups and the tooltip re-append themselves later in the frame, so they stay above.
    */
    void raiseFocusedWindow() {
        if (!pressStarted || captured == 0) return;

        // walk to the root: the press may have landed on a widget many levels down
        uint32_t root = captured;
        while (root != 0 && elements[root].parent != 0) root = elements[root].parent;
        if (root == 0 || root == tooltipElement) return;

        focusedWindow = root;
        setParent(root, 0); // re-append == raise
    }

    // Key edge, with GLFW's auto-repeat counting so held backspace keeps deleting.
    static bool keyPressed(int key) {
        const auto& current = InputManager::getCurrentState();
        auto it = current.keyStates.find(key);
        if (it == current.keyStates.end()) return false;
        if (it->second == GLFW_REPEAT) return true;
        if (it->second != GLFW_PRESS) return false;

        const auto& previous = InputManager::getPreviousState();
        auto prev = previous.keyStates.find(key);
        return prev == previous.keyStates.end() || prev->second != GLFW_PRESS;
    }

    // Fine-drag modifier: shift, held, scales a slider or drag's travel by style.fineDragScale.
    static bool fineModifierHeld() {
        const auto& keys = InputManager::getCurrentState().keyStates;
        for (int key : {GLFW_KEY_LEFT_SHIFT, GLFW_KEY_RIGHT_SHIFT}) {
            auto it = keys.find(key);
            if (it != keys.end() && (it->second == GLFW_PRESS || it->second == GLFW_REPEAT)) return true;
        }
        return false;
    }

    bool editingItem(uint64_t id) const {
        auto it = imItems.find(id);
        return it != imItems.end() && it->second.editing;
    }

    // Whether this id's typed value landed this frame. Frame-scoped rather than a consumed flag, so
    // a caller that never asks doesn't leave one set forever.
    bool justCommitted(uint64_t id) const { return id != 0 && id == committedID && committedFrame == frameCounter; }

    /*
    Hands the item the caret and seeds the typing buffer from the live value.

    The widget's display format is deliberately not reused: several are decorated ("min %.1f",
    "%.2f /s") or empty (the int wrappers), and would seed text that does not read back as a number.
    */
    void beginNumericEdit(uint64_t id, uint32_t item, float value) {
        std::string& buffer = inputBuffers[id];
        buffer.assign(64, '\0');
        snprintf(buffer.data(), buffer.size(), "%g", value);
        caretPos[id] = strnlen(buffer.data(), buffer.size());

        GUIItemSlot& slot = imItems[id];
        slot.editing = true;
        slot.editFresh = true;
        // hitTest only moves the caret onto a TextBox, and the element was still a slider when the
        // press landed — the type only changes next frame, so the focus has to be handed over here
        keyboardFocus = item;
        guiWantsKeyboard = true;
    }

    /*
    A slider or drag double-clicked into typing mode: its own slot, drawn as a text field until the
    value is committed with Enter or by clicking away. Escape leaves the bound value untouched.

    Runs on the widget's id, so the field lands in the same layout slot, keeps the same label, and
    inputTextImpl resolves it to the same element the slider was.
    */
    bool numericField(uint64_t id, std::string_view label, float* value, float min, float max) {
        std::string& buffer = inputBuffers[id];
        if (buffer.size() < 64) buffer.resize(64, '\0');

        // The field has no selection, so the first character typed stands in for replacing the lot —
        // a double click is usually the start of a fresh number. Backspace and the arrows edit it.
        if (imItems[id].editFresh && !InputManager::getCurrentState().charQueue.empty()) {
            buffer[0] = '\0';
            caretPos[id] = 0;
            imItems[id].editFresh = false;
        }

        inputTextImpl(label, {}, buffer.data(), buffer.size());
        uint32_t item = itemOf(id);

        bool cancelled = keyPressed(GLFW_KEY_ESCAPE);
        // losing the caret means the user clicked something else, which commits like every other field
        bool commit = keyPressed(GLFW_KEY_ENTER) || keyPressed(GLFW_KEY_KP_ENTER) || keyboardFocus != item;
        if (!cancelled && !commit) return false;

        imItems[id].editing = false;
        // The field's run lives on the element; a slider paints its value into a child instead and
        // never overwrites it, so what was typed would keep drawing under the fill from here on.
        clearText(item);
        // inputTextImpl latched this for every keystroke. The commit below is the edit edge callers
        // are told about, so leaving it set would fire a second one on the next click of the slider.
        imItems[id].editedWhileActive = false;
        if (keyboardFocus == item) {
            keyboardFocus = 0;
            guiWantsKeyboard = false;
        }
        if (cancelled) return false;

        // Accept only a parse that consumed something: an empty field, or a half-typed "-", leaves
        // the value alone rather than snapping it to zero.
        char* parseEnd = nullptr;
        float parsed = strtof(buffer.data(), &parseEnd);
        if (parseEnd == buffer.data()) return false;

        // min < max is the drags' own "bounded" test; the sliders always pass a real range
        *value = (min < max) ? glm::clamp(parsed, min, max) : parsed;
        committedID = id;
        committedFrame = frameCounter;
        return true;
    }

    bool inputTextImpl(std::string_view label, std::string_view hint, char* buffer, size_t bufferSize) {
        if (buffer == nullptr || bufferSize == 0) return false;
        uint64_t id = hashID(label, currentID());

        GUILayout* layout = getLayout(currentWindow());
        float width = (layout != nullptr && layout->nextItemWidth > 0.0f) ? layout->nextItemWidth : style.itemWidth;
        float height = getLineHeight() + style.framePaddingY * 2.0f;

        uint32_t item = acquireItem(id, GUIRect{.type = GUIType::Button | GUIType::TextBox,
                                                 .size = glm::vec2(width, height),
                                                 .color = style.frameBg,
                                                 .hoverColor = style.frameBgHover,
                                                 .pressedColor = style.frameBgHover,
                                                 .heldColor = style.frameBgHover});
        if (item == 0) return false;

        size_t length = strnlen(buffer, bufferSize - 1);
        size_t& caret = caretPos[id];
        caret = std::min(caret, length);

        bool changed = false;
        if (keyboardFocus == item) {
            // Text first: the char callback has already resolved layout and dead keys, so this is
            // what the user actually typed rather than which keys they pressed.
            for (uint32_t codepoint : InputManager::getCurrentState().charQueue) {
                if (codepoint < 32 || codepoint > 126 || length + 1 >= bufferSize) continue;
                memmove(buffer + caret + 1, buffer + caret, length - caret + 1);
                buffer[caret] = static_cast<char>(codepoint);
                caret++;
                length++;
                changed = true;
            }
            if (keyPressed(GLFW_KEY_BACKSPACE) && caret > 0) {
                memmove(buffer + caret - 1, buffer + caret, length - caret + 1);
                caret--;
                length--;
                changed = true;
            }
            if (keyPressed(GLFW_KEY_DELETE) && caret < length) {
                memmove(buffer + caret, buffer + caret + 1, length - caret);
                length--;
                changed = true;
            }
            if (keyPressed(GLFW_KEY_LEFT) && caret > 0) caret--;
            if (keyPressed(GLFW_KEY_RIGHT) && caret < length) caret++;
            if (keyPressed(GLFW_KEY_HOME)) caret = 0;
            if (keyPressed(GLFW_KEY_END)) caret = length;
            buffer[bufferSize - 1] = '\0';
        }

        bool empty = length == 0;
        std::string shown = empty && !hint.empty() ? std::string(hint) : std::string(buffer, length);
        setText(item, shown, empty && !hint.empty() ? style.hintText : style.text);
        glm::vec2 textPadding(6.0f, style.framePaddingY);
        if (GUITextRun* run = getTextMutable(item)) run->padding = textPadding;

        // Caret: a rule at the measured width of the text before it. Blinks so a focused-but-idle
        // field still reads as focused.
        // off the frame clock, not the hover clock: the caret must keep its rhythm while the cursor
        // wanders over other widgets
        bool blinkOn = std::fmod(std::chrono::duration<float>(frameTime - startTime).count(), style.caretBlink * 2.0f) < style.caretBlink;
        bool showCaret = keyboardFocus == item && blinkOn;
        glm::vec4 caretColor = showCaret ? style.caret : glm::vec4(0);
        float caretX = textPadding.x + measureText(std::string_view(buffer, caret)).x;
        acquireChild(hashID("##caret", id), item, GUIRect{.size = glm::vec2(1.0f, getLineHeight()),
                                                           .offset = glm::vec2(caretX, style.framePaddingY),
                                                           .color = caretColor,
                                                           .hoverColor = caretColor,
                                                           .pressedColor = caretColor,
                                                           .heldColor = caretColor,
                                                           .hitTestable = false,
                                                           .anchor = GUIAnchor::Top | GUIAnchor::Left});

        labelAfter(label);
        setLastItem(item, id);
        if (changed) imItems[id].editedWhileActive = true;
        return changed;
    }

    // Shared by colorEdit4 and colorPicker3: a live swatch, then one slider per channel.
    bool colorSwatchAndSliders(float* color, int channels) {
        glm::vec4 swatch(color[0], color[1], color[2], channels > 3 ? color[3] : 1.0f);
        float row = getLineHeight() + style.framePaddingY * 2.0f;
        acquireItem(nextAutoID("##swatch"), GUIRect{.size = glm::vec2(row * 2.0f, row),
                                                     .color = swatch,
                                                     .hoverColor = swatch,
                                                     .pressedColor = swatch,
                                                     .heldColor = swatch,
                                                     .hitTestable = false});

        static const char* names[4] = {"R##ch", "G##ch", "B##ch", "A##ch"};
        bool changed = false;
        for (int i = 0; i < channels; i++) {
            changed |= sliderFloat(names[i], &color[i], 0.0f, 1.0f, "%.3f");
        }
        return changed;
    }

    // Builds the deferred tooltip, last of all, so it lands at the tail of element 0's children
    // and therefore on top of every window and popup.
    void flushTooltip() {
        if (pendingTooltip.empty()) {
            if (tooltipElement != 0) {
                removeElement(tooltipElement);
                tooltipElement = 0;
            }
            return;
        }

        glm::vec2 padding(8.0f, 4.0f);
        glm::vec2 size = measureText(pendingTooltip) + padding * 2.0f;
        // offset below-right of the cursor, pulled back inside the screen when it would overflow
        glm::vec2 screen = elements[0].size;
        glm::vec2 pos = InputManager::getCurrentState().mousePos + glm::vec2(16.0f, 20.0f);
        pos = glm::min(pos, glm::max(screen - size - glm::vec2(4.0f), glm::vec2(0.0f)));

        GUIRect prototype{.size = size,
                          .offset = pos,
                          .color = style.tooltipBg,
                          .hoverColor = style.tooltipBg,
                          .pressedColor = style.tooltipBg,
                          .heldColor = style.tooltipBg,
                          .hitTestable = false,
                          .anchor = GUIAnchor::Top | GUIAnchor::Left};

        if (get(tooltipElement) == nullptr) {
            tooltipElement = addElement(prototype, 0);
        } else {
            applyPrototype(elements[tooltipElement], prototype);
            setParent(tooltipElement, 0); // re-append: topmost is whatever was added last
        }
        if (tooltipElement == 0) return;
        setText(tooltipElement, pendingTooltip, style.text);
        if (GUITextRun* run = getTextMutable(tooltipElement)) run->padding = padding;
    }

    // Measured against the viewport, not the container: the header is not content space.
    static glm::vec2 overflowOf(const GUILayout& layout, const GUIRect& container) {
        glm::vec2 visible = viewportSize(layout, container) - layout.style.padding * 2.0f;
        return glm::max(layout.contentExtent - visible, glm::vec2(0));
    }

    static void resetLayoutCursor(GUILayout& layout) {
        layout.penX = 0.0f;
        layout.rowTop = 0.0f;
        layout.rowHeight = 0.0f;
        layout.indent = 0.0f;
        layout.rowStarted = false;
        layout.sameLinePending = false;
        layout.nextItemWidth = 0.0f;
        layout.contentExtent = glm::vec2(0.0f);
        // scroll survives on purpose — a rebuild shouldn't jump the user back to the top
    }

    // Places one item and moves the cursor past it. Returns the offset from the CONTENT ROOT's top-left.
    static glm::vec2 advanceCursor(GUILayout& layout, glm::vec2 size) {
        if (layout.sameLinePending) {
            float gap = layout.sameLineSpacing < 0.0f ? layout.style.itemSpacing.x : layout.sameLineSpacing;
            layout.penX += gap;
            layout.sameLinePending = false;
        } else {
            if (layout.rowStarted) {
                layout.rowTop += layout.rowHeight + layout.style.itemSpacing.y;
                layout.rowHeight = 0.0f;
            }
            layout.penX = layout.indent;
            layout.rowStarted = true;
        }

        glm::vec2 pos(layout.penX, layout.rowTop);
        layout.penX += size.x;
        layout.rowHeight = std::max(layout.rowHeight, size.y);
        layout.contentExtent = glm::max(layout.contentExtent, pos + size);
        return pos;
    }

    /*
    The viewport/content-root pair every laid-out item hangs off. 
    Created on first use rather than in beginLayout, so a container that never gets content costs nothing.
    */
    uint32_t ensureContentRoot(uint32_t container, GUILayout& layout) {
        if (get(layout.contentRoot) != nullptr) return layout.contentRoot;

        GUIRect* c = get(container);
        if (c == nullptr) return 0;

        layout.viewport = addElement(GUIRect{.size = viewportSize(layout, *c),
                                             .offset = glm::vec2(0.0f, layout.style.headerHeight),
                                             .color = glm::vec4(0),
                                             .hoverColor = glm::vec4(0),
                                             .pressedColor = glm::vec4(0),
                                             .heldColor = glm::vec4(0),
                                             .hitTestable = false,
                                             .anchor = GUIAnchor::Top | GUIAnchor::Left},
                                     container);
        if (layout.viewport == 0) return 0;

        layout.contentRoot = addElement(GUIRect{.size = glm::vec2(0),
                                                .offset = layout.style.padding + layout.scroll,
                                                .color = glm::vec4(0),
                                                .hoverColor = glm::vec4(0),
                                                .pressedColor = glm::vec4(0),
                                                .heldColor = glm::vec4(0),
                                                .hitTestable = false,
                                                .clipsChildren = false,
                                                .anchor = GUIAnchor::Top | GUIAnchor::Left},
                                        layout.viewport);
        return layout.contentRoot;
    }

    static glm::vec2 viewportSize(const GUILayout& layout, const GUIRect& container) {
        return glm::max(container.size - glm::vec2(0.0f, layout.style.headerHeight), glm::vec2(0.0f));
    }

    void updateScroll() {
        float wheel = InputManager::getCurrentState().scroll.y;
        for (auto& [container, layout] : layouts) {
            GUIRect* c = get(container);
            if (c == nullptr) continue;

            if (layout.scrollable && wheel != 0.0f && hoveredElement != 0 && isAncestorOf(container, hoveredElement)) {
                layout.scroll.y += wheel * layout.style.scrollSpeed;
            }
            layout.scroll.y = glm::clamp(layout.scroll.y, -overflowOf(layout, *c).y, 0.0f);

            // the viewport tracks the container's size so a resize re-clips correctly; the content
            // root only ever carries the scroll offset
            if (GUIRect* view = get(layout.viewport)) {
                view->size = viewportSize(layout, *c);
                view->offset = glm::vec2(0.0f, layout.style.headerHeight);
            }
            if (GUIRect* root = get(layout.contentRoot)) root->offset = layout.style.padding + layout.scroll;
        }
    }

    // Moves a Resizeable's grips to the tail of its child list. setParent re-appends, so that is
    // the whole primitive; the collect-then-move split is because setParent rewrites nextSibling
    // and walking the chain while cutting it would drop the rest of the list.
    void raiseResizeHandles(uint32_t panel) {
        if (panel == 0 || panel >= elements.size() || !elements[panel].alive()) return;
        if (!(elements[panel].type & GUIType::Resizeable)) return;

        reorderScratch.clear();
        for (uint32_t c = elements[panel].firstChild; c != 0; c = elements[c].nextSibling) {
            if (elements[c].internalType & InternalType::ResizeableHandle) reorderScratch.push_back(c);
        }
        for (uint32_t handle : reorderScratch) setParent(handle, panel);
    }

    bool isAncestorOf(uint32_t maybeAncestor, uint32_t node) const {
        for (uint32_t n = node; n != 0; n = elements[n].parent) {
            if (n == maybeAncestor) return true;
        }
        return false;
    }

    // element 0 is the dead sentinel and doubles as the screen rect during layout
    std::vector<GUIRect> elements = {GUIRect{}};
    std::queue<uint32_t> freeSlots;

    // rect the live press started on; owns the press until the button comes back up
    uint32_t captured = 0;
    uint32_t clickedElement = 0;
    uint32_t hoveredElement = 0;
    // second click on the same rect inside style.doubleClickTime, resolved by hitTest
    uint32_t doubleClickedElement = 0;
    uint32_t lastClickElement = 0;
    std::chrono::steady_clock::time_point lastClickTime{};
    // rect the caret is in; survives across frames until another press moves it
    uint32_t keyboardFocus = 0;

    // published by hitTest, read by InputManager::tickInputState to gate the scene raycast,
    // camera controls and hotkeys
    bool guiWantsMouse = false;
    bool guiWantsKeyboard = false;

    // linear scan per dispatch — a handful of types, and the AND is cheaper than a per-bit table
    std::vector<GUITypeHandler> handlers;

    // per-element callbacks, the counterpart to the type-keyed `handlers`. Sparse: only the few
    // rects that actually do something get an entry, which is why it's a map and not a column
    // on GUIRect.
    std::unordered_map<uint32_t, GUIHook> actions;

    // Strings, keyed the same way and for the same reason: variable-length, and only some elements
    // have one. GUIRect stays a POD the quad path can memcpy. See GUITextRun.
    std::unordered_map<uint32_t, GUITextRun> texts;

    // Layout cursors, keyed by the container that owns them. Sparse for the same reason: only
    // panels lay their children out, and most elements are content.
    std::unordered_map<uint32_t, GUILayout> layouts;

    // Immediate-mode item table: hashed label -> the retained element standing in for it. The
    // reverse of the other side tables — keyed by id, not by element.
    std::unordered_map<uint64_t, GUIItemSlot> imItems;
    // dragInt's float accumulator: rounding the delta every frame instead would quantise any
    // speed below one pixel-per-frame straight to zero
    std::unordered_map<uint64_t, float> dragAccum;
    // sliderFloat's unclamped-at-the-ends drag value, armed by the press and moved by cursor travel
    std::unordered_map<uint64_t, float> sliderAccum;
    // caret offset per text field, in characters
    std::unordered_map<uint64_t, size_t> caretPos;
    // inputFloat's text buffer: the field edits characters, the caller holds a float
    std::unordered_map<uint64_t, std::string> inputBuffers;
    std::vector<uint64_t> idStack;
    std::vector<uint32_t> windowStack;
    std::vector<uint32_t> autoIDStack;
    // open combos, innermost last: {combo id, popup element}
    std::vector<std::pair<uint64_t, uint32_t>> comboStack;
    uint64_t frameCounter = 0;
    // per-scope counter behind nextAutoID, saved and reset by pushWindowScope
    uint32_t autoID = 0;
    // the widget most recently called, which every isItemXxx query refers to
    uint32_t lastItem = 0;
    uint64_t lastItemID = 0;
    // id whose typed value landed, and the frame it landed on. One slot: only the item holding the
    // caret can commit, and there is only ever one of those.
    uint64_t committedID = 0;
    uint64_t committedFrame = 0;
    // press owner as of last frame, so activation/deactivation edges are visible
    uint32_t capturedPrev = 0;
    // press *edge* this frame — what closes a popup clicked away from
    bool pressStarted = false;
    bool mouseDown = false;
    // when the cursor arrived on hoveredElement — the tooltip delay counts from here
    std::chrono::steady_clock::time_point hoverSince = std::chrono::steady_clock::now();
    // stamped once per beginFrame, so everything animating in a frame agrees on the time
    std::chrono::steady_clock::time_point frameTime = std::chrono::steady_clock::now();
    const std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    std::string pendingTooltip;
    uint32_t tooltipElement = 0;
    // root last pressed in — drives the title bar highlight and the z-order raise
    uint32_t focusedWindow = 0;
    // pending setNextWindowPos, consumed by the next beginWindow
    glm::vec2 nextWindowPos = glm::vec2(0);
    bool hasNextWindowPos = false;
    // pending setNextWindowSize, same contract
    glm::vec2 nextWindowSize = glm::vec2(0);
    bool hasNextWindowSize = false;

    // Per-window position/size/collapse, surviving the element. Keyed by window id; savedLayout is
    // the same thing read from disk, keyed by name because that is what is stable across runs.
    std::unordered_map<uint64_t, GUIWindowState> windowStates;
    std::unordered_map<std::string, GUIWindowState> savedLayout;

    GUIStyle style;

    GUIFont font;

    // reused per call, avoids a per-frame allocation
    std::vector<GPUGuiQuad> quadStaging;
    std::vector<uint32_t> scratch;
    // separate from `scratch`: raiseResizeHandles runs from addElement, which a handler can call
    // from inside a walk that is already using it
    std::vector<uint32_t> reorderScratch;
    uint32_t quadCount = 0;

    uint32_t GPUBufferIndex = 0xFFFFFFFF;
    uint32_t pipelineIndex = 0xFFFFFFFF;
    uint32_t atlasTextureIndex = 0xFFFFFFFF;
    uint32_t nearestSamplerIndex = 0xFFFFFFFF;
    uint32_t linearSamplerIndex = 0xFFFFFFFF;
};
