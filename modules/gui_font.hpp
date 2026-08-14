#pragma once
#include <glm/glm.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include <stb_truetype.h>

/*
Font atlas for the custom GUI: one 8-bit coverage bake, expanded to RGBA8 and uploaded once at
startup. Plain alpha.
*/

// Fixed at compile time because GUIRect's default uv has to name the white strip, and GUIRect
// exists long before any atlas does.
constexpr uint32_t GUI_ATLAS_PX = 512;
// Rows reserved at the top of the atlas, across its full width. The packer is handed the rest.
constexpr uint32_t GUI_ATLAS_WHITE_ROWS = 8;
// A sub-rect well inside the strip: 2px of margin all round, so neither filtering nor a mip can
// reach a glyph.
constexpr float GUI_WHITE_UV_LO = 2.0f / GUI_ATLAS_PX;
constexpr float GUI_WHITE_UV_HI = 6.0f / GUI_ATLAS_PX;

// Printable ASCII, 32..126. may widen this later for more characters if needed
constexpr uint32_t GUI_FIRST_CHAR = 32;
constexpr uint32_t GUI_CHAR_COUNT = 95;

constexpr float GUI_DEFAULT_FONT_PX = 15.0f;
constexpr uint32_t GUI_TAB_SPACES = 4;

// Everything in px, resolved at bake time. `offset` and `advance` are relative to the pen, which
// sits on the baseline — see GUIFont::ascent for getting there from the top of a line.
struct GUIGlyph {
    glm::vec2 uvMin = glm::vec2(0);
    glm::vec2 uvMax = glm::vec2(0);
    glm::vec2 offset = glm::vec2(0); // pen -> the glyph quad's top-left
    glm::vec2 size = glm::vec2(0);   // 0 for whitespace, which draws nothing but still advances
    float advance = 0.0f;
};

class GUIFont {

  public:

    GUIFont() { resetAtlas(); }

    // Bakes the first candidate that loads, and returns its path (empty if none did).
    std::string bakeFirstAvailable(const std::vector<std::string>& candidates, float pixelHeight) {
        for (const std::string& path : candidates) {
            if (bakeFromFile(path, pixelHeight)) return path;
        }
        return {};
    }

    bool bakeFromFile(const std::string& path, float pixelHeight) {
        std::vector<uint8_t> ttf;
        if (!readWholeFile(path, ttf)) return false;
        return bake(ttf, pixelHeight);
    }

    // looks for fonts, and fallbacks (windows paths only for now)
    static std::vector<std::string> defaultCandidates() {
        return {"fonts/gui.ttf",
                "C:/Windows/Fonts/consola.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "C:/Windows/Fonts/arial.ttf"};
    }

    bool loaded() const { return baked; }

    // RGBA8, GUI_ATLAS_PX square. Valid even when no font loaded
    const uint8_t* atlasPixels() const { return atlas.data(); }

    // The atlas is ~1 MB and dead the moment it reaches the GPU. Glyph metrics are what the
    // renderer actually keeps, and they survive this.
    void releasePixels() {
        atlas.clear();
        atlas.shrink_to_fit();
    }

    // px from the top of a line down to the baseline
    float ascent() const { return asc; }
    // px between one line's top and the next's — what a newline advances by
    float lineHeight() const { return lineH; }
    // advance of a space, the unit tab stops are counted in
    float spaceAdvance() const { return glyphs[' ' - GUI_FIRST_CHAR].advance; }

    // null for anything outside the baked range, including control characters
    const GUIGlyph* glyph(uint32_t codepoint) const {
        if (codepoint < GUI_FIRST_CHAR || codepoint >= GUI_FIRST_CHAR + GUI_CHAR_COUNT) return nullptr;
        return &glyphs[codepoint - GUI_FIRST_CHAR];
    }

    /*
    Bounding box of a run in px: the widest line by advance, and one lineHeight per line. Handles
    the same '\n' and '\t' the renderer does, so a measured box and the drawn glyphs agree.

    Required before layout can size anything to its label, which is why it lands with the text
    rather than with the layout engine that needs it.
    */
    glm::vec2 measure(std::string_view text) const {
        float widest = 0.0f;
        float x = 0.0f;
        uint32_t lines = 1;
        for (char raw : text) {
            unsigned char c = static_cast<unsigned char>(raw);
            if (c == '\n') {
                widest = std::max(widest, x);
                x = 0.0f;
                lines++;
                continue;
            }
            if (c == '\t') {
                x += spaceAdvance() * GUI_TAB_SPACES;
                continue;
            }
            if (const GUIGlyph* g = glyph(c)) x += g->advance;
        }
        return glm::vec2(std::max(widest, x), lines * lineH);
    }

  private:

    static bool readWholeFile(const std::string& path, std::vector<uint8_t>& out) {
        FILE* f = nullptr;
        if (fopen_s(&f, path.c_str(), "rb") != 0 || f == nullptr) return false;
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (size <= 0) {
            fclose(f);
            return false;
        }
        out.resize(static_cast<size_t>(size));
        size_t read = fread(out.data(), 1, out.size(), f);
        fclose(f);
        return read == out.size();
    }

    // White strip + fully transparent everywhere else. Called before every bake so a failed
    // attempt can't leave half a font behind for the next candidate.
    void resetAtlas() {
        atlas.assign(static_cast<size_t>(GUI_ATLAS_PX) * GUI_ATLAS_PX * 4, 0);
        for (uint32_t y = 0; y < GUI_ATLAS_WHITE_ROWS; y++) {
            uint8_t* row = atlas.data() + static_cast<size_t>(y) * GUI_ATLAS_PX * 4;
            std::fill(row, row + static_cast<size_t>(GUI_ATLAS_PX) * 4, uint8_t(255));
        }
        glyphs.fill(GUIGlyph{});
        baked = false;
    }

    bool bake(const std::vector<uint8_t>& ttf, float pixelHeight) {
        resetAtlas();

        stbtt_fontinfo info;
        int fontOffset = stbtt_GetFontOffsetForIndex(ttf.data(), 0);
        if (fontOffset < 0 || !stbtt_InitFont(&info, ttf.data(), fontOffset)) return false;

        float scale = stbtt_ScaleForPixelHeight(&info, pixelHeight);
        int ascentUnits = 0, descentUnits = 0, lineGapUnits = 0;
        stbtt_GetFontVMetrics(&info, &ascentUnits, &descentUnits, &lineGapUnits);
        asc = ascentUnits * scale;
        // descent is negative, so this is ascent + |descent| + gap. Rounded up to keep successive lines on whole pixels
        lineH = std::ceil((ascentUnits - descentUnits + lineGapUnits) * scale);

        /*
        Pack into the atlas below the reserved rows.

        Packing 8-bit and expanding afterwards rather than packing straight into RGBA

        The stride MUST stay equal to the width. PackBegin clears its bitmap with a flat
        memset(pixels, 0, pw*ph) that ignores the stride, so a window narrower than its stride would
        run off the end of the buffer. Full-width rows starting at an offset is the one shape that
        is safe, and it is exactly the shape a reserved strip at the top gives.
        */
        std::vector<uint8_t> coverage(static_cast<size_t>(GUI_ATLAS_PX) * GUI_ATLAS_PX, 0);
        uint8_t* packOrigin = coverage.data() + static_cast<size_t>(GUI_ATLAS_WHITE_ROWS) * GUI_ATLAS_PX;
        int packRows = static_cast<int>(GUI_ATLAS_PX - GUI_ATLAS_WHITE_ROWS);

        stbtt_pack_context pack;
        if (!stbtt_PackBegin(&pack, packOrigin, GUI_ATLAS_PX, packRows, GUI_ATLAS_PX, 1, nullptr)) return false;
        // 1x1: the UI draws glyphs at their baked size and pixel-aligned, so oversampling would
        // cost atlas area and blur for nothing.
        stbtt_PackSetOversampling(&pack, 1, 1);

        std::array<stbtt_packedchar, GUI_CHAR_COUNT> packed{};
        int ok = stbtt_PackFontRange(&pack, ttf.data(), 0, pixelHeight, GUI_FIRST_CHAR, GUI_CHAR_COUNT, packed.data());
        stbtt_PackEnd(&pack);
        if (!ok) return false;

        // coverage -> RGBA8. White throughout, alpha carries the shape, so GPUGuiQuad::color tints
        // the glyph and the existing alpha-blend pipeline composites it with no shader change.
        for (uint32_t y = GUI_ATLAS_WHITE_ROWS; y < GUI_ATLAS_PX; y++) {
            for (uint32_t x = 0; x < GUI_ATLAS_PX; x++) {
                size_t i = static_cast<size_t>(y) * GUI_ATLAS_PX + x;
                uint8_t* px = atlas.data() + i * 4;
                px[0] = px[1] = px[2] = 255;
                px[3] = coverage[i];
            }
        }

        constexpr float inv = 1.0f / GUI_ATLAS_PX;
        for (uint32_t i = 0; i < GUI_CHAR_COUNT; i++) {
            const stbtt_packedchar& pc = packed[i];
            // shift the packer's y back into full-atlas space
            float y0 = static_cast<float>(pc.y0 + GUI_ATLAS_WHITE_ROWS);
            float y1 = static_cast<float>(pc.y1 + GUI_ATLAS_WHITE_ROWS);

            GUIGlyph& g = glyphs[i];
            g.uvMin = glm::vec2(pc.x0 * inv, y0 * inv);
            g.uvMax = glm::vec2(pc.x1 * inv, y1 * inv);
            g.offset = glm::vec2(pc.xoff, pc.yoff);
            g.size = glm::vec2(static_cast<float>(pc.x1 - pc.x0), static_cast<float>(pc.y1 - pc.y0));
            g.advance = pc.xadvance;
        }

        baked = true;
        return true;
    }

    std::vector<uint8_t> atlas;
    std::array<GUIGlyph, GUI_CHAR_COUNT> glyphs{};
    float asc = 0.0f;
    float lineH = 0.0f;
    bool baked = false;
};
