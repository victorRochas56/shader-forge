#pragma once
#include <cmath>
#include <limits>
#include <unordered_map>
#include "scene.hpp"
#include "structs.hpp"
#include "utils.hpp"
#include "gizmo.hpp"
#include "serializable_interface.h"

namespace BuildingPiece {

    enum Type {
        WALL,
        FLOORWALL,
        WINDOW,
        FLOORWINDOW,
        ENTRANCE,
        ROOF,
        ROOFWINDOW,
        LINKED, // span pieces, only placed through another piece's link
        NONE
    };
    // Display names indexed by Type,> NONE last so a picker can offer "unset" as its final entry.
    inline constexpr const char* typeNames[] = {
        "Wall",
        "Floor Wall",
        "Window",
        "Floor Window",
        "Entrance",
        "Roof",
        "Roof Window",
        "Linked",
        "<none>"};
    inline constexpr int typeCount = static_cast<int>(NONE) + 1;
    inline const char* typeName(Type type) { return typeNames[static_cast<int>(type)]; }

    // Reverse of typeName, for reading a saved piece back. Names rather than the enum's values go
    // in the file, so reordering the enum can't silently refile every piece in an old scene.
    // Anything unrecognised reads as NONE, which has no bucket.
    inline Type typeFromName(const std::string& name) {
        for (int i = 0; i < typeCount; i++) {
            if (name == typeNames[i]) return static_cast<Type>(i);
        }
        return NONE;
    }

    // Where a link puts its target, relative to the piece it is on. A link is a rule: if this piece
    // is in a cell, the cell it points at holds one of the pieces linked there.
    enum class LinkedPos
    {
        VERT_ABOVE, // element is above
        VERT_BELOW, // is below
        VERT_BOTH, // both above and below
        SIDE_POS, // after: to the right, seen from outside the wall
        SIDE_NEG, // before: to the left
        SIDE_BOTH, // connects on both sides
        ABOVE_BEFORE, // one row up, one slot back
        ABOVE_AFTER,
        BELOW_BEFORE,
        BELOW_AFTER
    };
    // Display and save names, indexed by LinkedPos.
    inline constexpr const char* linkPosNames[] = {
        "Above",
        "Below",
        "Above & Below",
        "After",
        "Before",
        "Before & After",
        "Above & Before",
        "Above & After",
        "Below & Before",
        "Below & After"};
    inline constexpr int linkPosCount = static_cast<int>(LinkedPos::BELOW_AFTER) + 1;
    inline const char* linkPosName(LinkedPos pos) { return linkPosNames[static_cast<int>(pos)]; }
    inline bool linkPosFromName(const std::string& name, LinkedPos& pos) {
        for (int i = 0; i < linkPosCount; i++) {
            if (name == linkPosNames[i]) { pos = static_cast<LinkedPos>(i); return true; }
        }
        // The two-way links' names before they read as "both".
        if (name == "Above or Below") { pos = LinkedPos::VERT_BOTH; return true; }
        if (name == "Either Side")    { pos = LinkedPos::SIDE_BOTH; return true; }
        return false;
    }

    // A cell offset: dx along the walk, dy up the wall.
    struct Offset { int dx; int dy; };
    inline bool operator==(Offset a, Offset b) { return a.dx == b.dx && a.dy == b.dy; }

    // Whether a link at `pos` reaches the cell `at`. Two-way links reach both ways.
    inline bool reaches(LinkedPos pos, Offset at) {
        switch (pos) {
        case LinkedPos::VERT_ABOVE:   return at == Offset{0, 1};
        case LinkedPos::VERT_BELOW:   return at == Offset{0, -1};
        case LinkedPos::VERT_BOTH:    return at.dx == 0 && at.dy != 0;
        case LinkedPos::SIDE_POS:     return at == Offset{1, 0};
        case LinkedPos::SIDE_NEG:     return at == Offset{-1, 0};
        case LinkedPos::SIDE_BOTH:    return at.dy == 0 && at.dx != 0;
        case LinkedPos::ABOVE_BEFORE: return at == Offset{-1, 1};
        case LinkedPos::ABOVE_AFTER:  return at == Offset{1, 1};
        case LinkedPos::BELOW_BEFORE: return at == Offset{-1, -1};
        case LinkedPos::BELOW_AFTER:  return at == Offset{1, -1};
        }
        return false;
    }

    // The one-way link reaching exactly `at`. The centre, which none reaches, reads as After.
    inline LinkedPos linkPosAt(Offset at) {
        if (at.dy > 0) return at.dx < 0 ? LinkedPos::ABOVE_BEFORE : at.dx > 0 ? LinkedPos::ABOVE_AFTER : LinkedPos::VERT_ABOVE;
        if (at.dy < 0) return at.dx < 0 ? LinkedPos::BELOW_BEFORE : at.dx > 0 ? LinkedPos::BELOW_AFTER : LinkedPos::VERT_BELOW;
        return at.dx < 0 ? LinkedPos::SIDE_NEG : LinkedPos::SIDE_POS;
    }

    // Target held by type + key, like groundKey, so re-registering doesn't break it and spans can loop.
    struct LinkedElement {
        Type type;
        std::string key;
        LinkedPos linkPos;
    };

    struct Element {
        Type type;
        // true for elements that are used as padding, every building set should have at least one, used to fill gaps
        bool resizeable;
        std::vector<std::string> templateKeys;
        float width;
        float height;
        // Ground-floor counterpart: one template key of the FLOOR* element that stands in for this
        // piece on the bottom row, so a column reads as the same piece all the way down. Stored as a
        // key rather than an index because the buckets are re-registered and erased under it. Empty
        // means unpaired — the bottom row then takes whatever its bucket offers.
        std::string groundKey;
        // Rules for the neighbouring cells. A cell some link reaches holds one of the pieces linked
        // there, picked at random.
        std::vector<LinkedElement> linkedElements;
    };

    // A closed plan in XZ, either winding. Edge i runs from corner i to corner i + 1, the last closing back to 0.
    struct Footprint {
        std::vector<glm::vec2> corners;

        int size() const { return static_cast<int>(corners.size()); }
        glm::vec2 corner(int i) const { const int n = size(); return corners[((i % n) + n) % n]; }

        // Two corners are a single wall, not a doubled-back pair.
        int edgeCount() const { return size() >= 3 ? size() : (size() == 2 ? 1 : 0); }

        // Shoelace. Positive is the winding whose walls face out as laid.
        float signedArea() const {
            float twice = 0.0f;
            for (int i = 0; i < size(); i++) {
                const glm::vec2 a = corner(i), b = corner(i + 1);
                twice += a.x * b.y - b.x * a.y;
            }
            return twice * 0.5f;
        }
    };

    // The types whose elements can stand in for `type` on the ground floor. Linked pieces may use either.
    inline std::vector<Type> groundFloorTypes(Type type) {
        switch (type) {
        case WALL:   return {FLOORWALL};
        case WINDOW: return {FLOORWINDOW};
        case LINKED: return {FLOORWINDOW, FLOORWALL};
        default:     return {};
        }
    }

}

class BuildingGen : public ISerializable
{

    std::vector<BuildingPiece::Element> walls;
    std::vector<BuildingPiece::Element> floorWalls;
    std::vector<BuildingPiece::Element> windows;
    std::vector<BuildingPiece::Element> floorWindows;
    std::vector<BuildingPiece::Element> entrances;
    std::vector<BuildingPiece::Element> roofs;
    std::vector<BuildingPiece::Element> roofWindows;
    std::vector<BuildingPiece::Element> linked;

    std::vector<uint32_t> spawnedNodes;

 public:
    BuildingGen() : ISerializable("BuildingGen") {}

    // Rolls for every wall derive from this, so a rebuild only changes the edges that changed.
    uint32_t seed = 1;

    // The bucket a type registers into; nullptr for NONE, so callers can treat "unset" as a miss
    // rather than having to test the enum themselves.
    std::vector<BuildingPiece::Element>* elementsFor(BuildingPiece::Type type)
    {
        switch (type)
        {
        case BuildingPiece::WALL:       return &walls;
        case BuildingPiece::FLOORWALL:  return &floorWalls;
        case BuildingPiece::WINDOW:     return &windows;
        case BuildingPiece::FLOORWINDOW:return &floorWindows;
        case BuildingPiece::ENTRANCE:   return &entrances;
        case BuildingPiece::ROOF:       return &roofs;
        case BuildingPiece::ROOFWINDOW: return &roofWindows;
        case BuildingPiece::LINKED:     return &linked;
        default:                        return nullptr;
        }
    }
    const std::vector<BuildingPiece::Element>* elementsFor(BuildingPiece::Type type) const
    {
        return const_cast<BuildingGen*>(this)->elementsFor(type);
    }

    // Registers a set of interchangeable templates as one piece, measured from their own bounds so
    // the walk can size a run without going back to the scene. Every key has to name a template —
    // the sizes would be meaningless otherwise, and a piece that can't be placed is worse than none
    // at all. Any key already registered under this type is taken off the element it was on: a
    // template is one piece's variant, not two, so re-registering re-measures instead of duplicating.
    bool registerElement(std::vector<std::string> templateKeys, BuildingPiece::Type type, bool resizeable, Scene& scene)
    {
        std::vector<BuildingPiece::Element>* bucket = elementsFor(type);
        if (!bucket || templateKeys.empty()) return false;

        // Variants share one slot, so the piece measures as the largest of them — reserving less
        // would let a bigger variant overlap whatever the walk put beside it. Placement stretches a
        // variant to the slot, so they are best authored at matching sizes.
        BuildingPiece::Element element{type, resizeable, templateKeys, 0.0f, 0.0f};
        for (const std::string& templateKey : templateKeys) {
            auto it = scene.templates.find(templateKey);
            if (it == scene.templates.end()) return false;

            // Pieces run along Z and stack along Y, the convention walkWidth reads back.
            glm::vec3 extent = it->second.bboxMax - it->second.bboxMin;
            element.width  = std::max(element.width, extent.z);
            element.height = std::max(element.height, extent.y);
        }

        auto sharesAKey = [&](const BuildingPiece::Element& existing) {
            for (const std::string& key : existing.templateKeys) {
                if (std::find(templateKeys.begin(), templateKeys.end(), key) != templateKeys.end()) return true;
            }
            return false;
        };
        // Carry the ground-floor pairing and links over from the element being replaced — re-measuring
        // a piece shouldn't silently unpair it.
        for (const BuildingPiece::Element& existing : *bucket) {
            if (sharesAKey(existing)) {
                element.groundKey = existing.groundKey;
                element.linkedElements = existing.linkedElements;
                break;
            }
        }
        std::erase_if(*bucket, sharesAKey);
        bucket->push_back(element);
        return true;
    }

    // The element in `type`'s bucket carrying `key` as one of its variants, or nullptr. Pairings are
    // stored as keys, so this is what resolves one — a pair whose element was dropped reads as unpaired.
    const BuildingPiece::Element* findElement(BuildingPiece::Type type, const std::string& key) const
    {
        const std::vector<BuildingPiece::Element>* bucket = elementsFor(type);
        if (!bucket || key.empty()) return nullptr;
        for (const BuildingPiece::Element& element : *bucket) {
            if (std::find(element.templateKeys.begin(), element.templateKeys.end(), key) != element.templateKeys.end()) {
                return &element;
            }
        }
        return nullptr;
    }
    BuildingPiece::Element* findElement(BuildingPiece::Type type, const std::string& key)
    {
        return const_cast<BuildingPiece::Element*>(static_cast<const BuildingGen&>(*this).findElement(type, key));
    }

    // The ground-floor piece `element` is paired with, or nullptr.
    const BuildingPiece::Element* findGround(const BuildingPiece::Element& element) const
    {
        for (BuildingPiece::Type type : BuildingPiece::groundFloorTypes(element.type)) {
            if (const BuildingPiece::Element* paired = findElement(type, element.groundKey)) return paired;
        }
        return nullptr;
    }

    // Links two registered pieces. Duplicates are refused so they can't skew the random pick.
    bool addLink(BuildingPiece::Type type, const std::string& key, const BuildingPiece::LinkedElement& link)
    {
        BuildingPiece::Element* element = findElement(type, key);
        if (!element || !findElement(link.type, link.key)) return false;
        for (const BuildingPiece::LinkedElement& existing : element->linkedElements) {
            if (existing.type == link.type && existing.key == link.key && existing.linkPos == link.linkPos) return false;
        }
        element->linkedElements.push_back(link);
        return true;
    }

/* ============= Links ================== */

    // Same registered piece, whichever copy: pieces are told apart by their templates, so a pad
    // stretched from the wall still reads as the wall.
    static bool samePiece(const BuildingPiece::Element& a, const BuildingPiece::Element& b) {
        return a.type == b.type && a.templateKeys == b.templateKeys;
    }
    static bool lists(const std::vector<const BuildingPiece::Element*>& options, const BuildingPiece::Element& piece) {
        for (const BuildingPiece::Element* option : options) {
            if (samePiece(*option, piece)) return true;
        }
        return false;
    }

    // The registered piece a copy came from, or nullptr if it was dropped since.
    const BuildingPiece::Element* registered(const BuildingPiece::Element& piece) const {
        return piece.templateKeys.empty() ? nullptr : findElement(piece.type, piece.templateKeys[0]);
    }

    // What `element`'s links put in the cell `at`. Empty when none reach it, which leaves that cell
    // free. Links to dropped pieces are skipped.
    std::vector<const BuildingPiece::Element*> linkOptions(const BuildingPiece::Element& element, BuildingPiece::Offset at) const
    {
        std::vector<const BuildingPiece::Element*> options;
        for (const BuildingPiece::LinkedElement& link : element.linkedElements) {
            if (!BuildingPiece::reaches(link.linkPos, at)) continue;
            if (const BuildingPiece::Element* target = findElement(link.type, link.key)) options.push_back(target);
        }
        return options;
    }

/* ============= Chains along the walk ================== */

    enum { CHAIN_BACK, CHAIN_FORWARD };
    static BuildingPiece::Offset chainStep(int side) { return {side == CHAIN_FORWARD ? 1 : -1, 0}; }

    // Shortest finish of a chain out of each registered piece, per direction, in width or in cells.
    // Infinite if it can't end.
    struct ChainTable {
        bool cells;
        std::unordered_map<const BuildingPiece::Element*, float> tail[2];
    };

    static float extent(const BuildingPiece::Element& element, const ChainTable& table) {
        return table.cells ? 1.0f : element.width;
    }

    static float finishFrom(const ChainTable& table, int side, const BuildingPiece::Element* piece) {
        auto it = table.tail[side].find(piece);
        return it != table.tail[side].end() ? it->second : std::numeric_limits<float>::infinity();
    }

    // Cheapest finish through any of `options`; zero when there's nothing to continue into.
    static float cheapestFinish(const std::vector<const BuildingPiece::Element*>& options, int side, const ChainTable& table) {
        float finish = options.empty() ? 0.0f : std::numeric_limits<float>::infinity();
        for (const BuildingPiece::Element* option : options) finish = std::min(finish, finishFrom(table, side, option));
        return finish;
    }

    ChainTable buildChainTable(bool cells) const
    {
        using namespace BuildingPiece;
        const float inf = std::numeric_limits<float>::infinity();

        std::vector<const Element*> pieces;
        for (int t = 0; t < NONE; t++) {
            for (const Element& element : *elementsFor(static_cast<Type>(t))) pieces.push_back(&element);
        }

        ChainTable table{cells};
        for (int side : {CHAIN_BACK, CHAIN_FORWARD}) {
            std::vector<std::vector<const Element*>> onward;
            onward.reserve(pieces.size());
            for (const Element* piece : pieces) onward.push_back(linkOptions(*piece, chainStep(side)));

            auto& tail = table.tail[side];
            for (const Element* piece : pieces) tail[piece] = inf;

            // Bellman-Ford; positive extents mean pieces.size() passes settle it.
            for (size_t pass = 0; pass < pieces.size(); pass++) {
                bool changed = false;
                for (size_t i = 0; i < pieces.size(); i++) {
                    const float own = extent(*pieces[i], table);
                    if (own <= 0.0f) continue; // would never advance
                    const float finish = own + cheapestFinish(onward[i], side, table);
                    if (finish < tail[pieces[i]]) {
                        tail[pieces[i]] = finish;
                        changed = true;
                    }
                }
                if (!changed) break;
            }
        }
        return table;
    }

    // Shortest span `seed` can lay: itself plus the cheapest way out of each side.
    float shortestSpan(const BuildingPiece::Element& seed, const ChainTable& table) const
    {
        return extent(seed, table)
            + cheapestFinish(linkOptions(seed, chainStep(CHAIN_BACK)), CHAIN_BACK, table)
            + cheapestFinish(linkOptions(seed, chainStep(CHAIN_FORWARD)), CHAIN_FORWARD, table);
    }

    // Follows links out one side, picking randomly among pieces that can still finish in `budget`. Returns the extent laid.
    float rollChain(std::vector<const BuildingPiece::Element*> options, int side, const ChainTable& table, float budget,
                    std::vector<BuildingPiece::Element>& chain) const
    {
        const float eps = 0.001f;
        float used = 0.0f;
        std::vector<const BuildingPiece::Element*> fitting;
        while (!options.empty()) {
            fitting.clear();
            for (const BuildingPiece::Element* option : options) {
                if (finishFrom(table, side, option) <= budget - used + eps) fitting.push_back(option);
            }
            if (fitting.empty()) break; // caller checked the shortest finish, so only a guard

            const BuildingPiece::Element* next = fitting.size() == 1 ? fitting[0] : fitting[randRange(0, static_cast<int>(fitting.size()) - 1)];
            chain.push_back(*next);
            used += extent(*next, table);
            options = linkOptions(*next, chainStep(side));
        }
        return used;
    }

    // One span around `seed`, at most `budget` long, back-most piece first. Empty if it can't fit.
    // Unlinked seeds come back alone without touching rand, so link-free sets build as before.
    std::vector<BuildingPiece::Element> rollSpan(const BuildingPiece::Element& seed, const ChainTable& table, float budget) const
    {
        using namespace BuildingPiece;
        const float eps = 0.001f;
        const std::vector<const Element*> backOptions = linkOptions(seed, chainStep(CHAIN_BACK));
        const std::vector<const Element*> forwardOptions = linkOptions(seed, chainStep(CHAIN_FORWARD));
        const float backFinish = cheapestFinish(backOptions, CHAIN_BACK, table);

        float used = extent(seed, table);
        if (used + backFinish + cheapestFinish(forwardOptions, CHAIN_FORWARD, table) > budget + eps) return {};

        // Forward rolls first, holding back the back side's shortest finish.
        std::vector<Element> before, after;
        used += rollChain(forwardOptions, CHAIN_FORWARD, table, budget - used - backFinish, after);
        rollChain(backOptions, CHAIN_BACK, table, budget - used, before);

        std::vector<Element> span(before.rbegin(), before.rend());
        span.push_back(seed);
        span.insert(span.end(), after.begin(), after.end());
        return span;
    }

    // Up to `count` distinct elements from a bucket, in random order. Sampling without replacement,
    // so a variety count can't spend two of its picks on the same piece; a count past what the
    // bucket holds just empties it.
    std::vector<BuildingPiece::Element> pickDistinct(const std::vector<BuildingPiece::Element>& bucket, uint32_t count)
    {
        std::vector<uint32_t> pool(bucket.size());
        for(uint32_t i = 0; i < pool.size(); i++) pool[i] = i;

        std::vector<BuildingPiece::Element> picked;
        while(picked.size() < count && !pool.empty()) {
            uint32_t at = static_cast<uint32_t>(randRange(0, static_cast<int>(pool.size()) - 1));
            picked.push_back(bucket[pool[at]]);
            pool.erase(pool.begin() + at);
        }
        return picked;
    }

    // Integer hash (lowbias32) — rand's LCG gives near-identical streams for nearby seeds.
    static uint32_t mixSeed(uint32_t x) {
        x ^= x >> 16; x *= 0x7feb352dU;
        x ^= x >> 15; x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    }

    // A wall along every edge of `plan`, replacing the last build. entrances[i] cuts an opening into edge i.
    // Mirrored walls are symmetric about their centre; otherwise each is walked end to end.
    void makeBuilding(const BuildingPiece::Footprint& plan, float height, uint32_t windowCount,
                      const std::vector<bool>& entrances, float cornerInset, bool mirrored, Scene& scene)
    {
        using namespace BuildingPiece;

        for(uint32_t node : spawnedNodes) {
            scene.sceneGraph.removeNode(node);
        }
        spawnedNodes.clear(); // the indices are free for reuse now; keeping them would kill the new run

        const int edges = plan.edgeCount();
        if(height <= 0.0f || edges == 0) return;

        // A clockwise plan walks each edge backwards so walls still face out; edge indices stay as drawn.
        const bool reversed = plan.signedArea() < 0.0f;

        for(int edge = 0; edge < edges; edge++) {
            // One wall per corner: odd edges give up both ends; on an odd count the last edge also yields corner 0.
            float headInset = (edge % 2 == 1) ? cornerInset : 0.0f;
            float tailInset = (edge % 2 == 1 || (edges > 1 && edges % 2 == 1 && edge == edges - 1)) ? cornerInset : 0.0f;

            glm::vec2 from = plan.corner(edge), to = plan.corner(edge + 1);
            if(reversed) {
                std::swap(from, to);
                std::swap(headInset, tailInset);
            }

            const glm::vec2 run = to - from;
            float length = glm::length(run);
            if(length <= 0.0f) continue; // a collapsed edge has no wall to build
            const glm::vec2 direction = run / length;

            length -= headInset + tailInset;
            if(length <= 0.0f) continue; // the inset ate the whole side
            const glm::vec2 start = from + direction * headInset;

            // Lays the run's +Z along the edge, putting its +X on the outward side.
            const float yaw = std::atan2(direction.x, direction.y);

            srand(mixSeed(seed ^ mixSeed(static_cast<uint32_t>(edge) + 1u)));
            makeWall(glm::vec3(start.x, 0.0f, start.y), yaw, glm::vec2(length, height), windowCount,
                     edge < static_cast<int>(entrances.size()) && entrances[edge], mirrored, scene);
        }
    }

    // One wall run: pieces laid along +Z from `origin`, the whole run turned by `yaw` about it.
    // Additive on purpose — makeBuilding clears the last building once, before its first wall.
    void makeWall(glm::vec3 origin, float yaw, glm::vec2 span, uint32_t windowCount, bool withEntrance, bool mirrored, Scene& scene)
    {
        using namespace BuildingPiece;

        // A set gets registered a piece at a time, so a run takes whatever is there and stands in
        // for the rest. The wall is the one piece it can't do without: it pads every gap the walk
        // can't fill with something else.
        if(walls.empty()) return;

        Element wall = walls[randRange(0, static_cast<int>(walls.size()) - 1)];
        Element floorWall = floorWalls.empty() ? wall : floorWalls[randRange(0, static_cast<int>(floorWalls.size()) - 1)];
        Element window_floor = floorWindows.empty() ? floorWall : floorWindows[randRange(0, static_cast<int>(floorWindows.size()) - 1)];

        // The row draws from several distinct windows rather than one, so a wide wall isn't the same
        // opening repeated. walkWidth reserves room for each, so all of them show up in the run.
        std::vector<Element> elements = {wall};
        for(const Element& window : pickDistinct(windows, windowCount)) elements.push_back(window);

        // The walk lays chains by width; the rows grown above it lay them by cell, since a slot
        // keeps the width the walk gave it.
        const ChainTable widths = buildChainTable(false);
        const ChainTable cells = buildChainTable(true);

        // A mirrored wall is walked to its centre and reflected; otherwise the walk covers the whole width.
        std::vector<Element> seq = walkWidth(elements, wall, mirrored ? span.x * 0.5f : span.x, widths);
        std::vector<Element> floorSeq;
        for(Element& el : seq) {
            // Each piece hands its slot to the ground-floor element it was paired with, so a column
            // reads as the same window all the way down. Unpaired pieces fall back to the random
            // pick from the bucket, which is all this did before pairings existed.
            const Element* paired = findGround(el);
            Element floorEl = paired ? *paired : (el.type == WINDOW ? window_floor : floorWall);
            // Widths come from the row above so the columns line up — the stretched pad included,
            // which has no width of its own to keep.
            floorEl.width = el.width;
            floorSeq.push_back(floorEl);
        }

        // Ground pieces reflect by the blocks of the row they stand under, having no links of their own.
        if(mirrored) mirror(floorSeq, spanBlocks(seq));
        // Only the faces that asked for one, and only if the set has one to cut in — a face with
        // nothing to place comes out solid rather than holed.
        if(withEntrance && !entrances.empty()) {
            Element entrance = entrances[randRange(0, static_cast<int>(entrances.size()) - 1)];
            placeEntrance(entrance, floorWall, floorSeq, span);
        }

        // Rows go up until the wall reaches span.y — the top one overshoots rather than leaving
        // the wall short. A row is as tall as its tallest piece; the ground row has its own height,
        // since ground pieces are often taller than the rest.
        float rowHeight = 0.0f;
        for(const Element& element : seq) rowHeight = std::max(rowHeight, element.height);
        if(rowHeight <= 0.0f) return; // unmeasured pieces would never advance the stack
        float floorHeight = 0.0f;
        for(const Element& element : floorSeq) floorHeight = std::max(floorHeight, element.height);
        if(floorHeight <= 0.0f) floorHeight = rowHeight;

        uint32_t rowCount = 0;
        for(float y = floorHeight; y < span.y; y += rowHeight) rowCount++;
        // Grown before any reflection, so a mirrored wall runs the same rules on both sides.
        std::vector<std::vector<Element>> rows = growRows(seq, wall, rowCount, cells);
        if(mirrored) {
            for(std::vector<Element>& row : rows) mirror(row, spanBlocks(row));
        }

        placeElements(floorSeq, origin, yaw, scene);
        float y = floorHeight;
        for(const std::vector<Element>& row : rows) {
            placeElements(row, origin + glm::vec3(0.0f, y, 0.0f), yaw, scene);
            y += rowHeight;
        }
    }

/* ============= Rows above the walk ================== */

    // The walked row and every row above it, grown bottom up like a cellular automaton: each cell
    // comes from the links of its settled neighbours. Widths stay the walked row's, so the columns
    // line up. Empty when the wall has no room above the ground row.
    std::vector<std::vector<BuildingPiece::Element>> growRows(const std::vector<BuildingPiece::Element>& walked,
        const BuildingPiece::Element& fill, uint32_t rowCount, const ChainTable& cells) const
    {
        using namespace BuildingPiece;

        if(rowCount == 0) return {};
        std::vector<std::vector<Element>> rows = {walked};
        for(uint32_t r = 1; r < rowCount; r++) {
            const std::vector<Element>& lower = rows.back();
            std::vector<Element> row;
            row.reserve(lower.size());
            for(size_t c = 0; c < lower.size(); c++) row.push_back(growCell(lower, row, c, fill, cells));
            rows.push_back(std::move(row));
        }
        return rows;
    }

    // One cell of a growing row, from its settled neighbours: the cell before it, the one below,
    // and the two below it diagonally. A link aimed here from any of them binds the cell to that
    // link's options — several, to what they all allow — and the pick is random among those that
    // fit the row and whose own links back at those neighbours agree. Where no link reaches, the
    // column carries on with the piece below, or the fill if that piece can't sit here.
    BuildingPiece::Element growCell(const std::vector<BuildingPiece::Element>& lower, const std::vector<BuildingPiece::Element>& row,
                                    size_t c, const BuildingPiece::Element& fill, const ChainTable& cells) const
    {
        using namespace BuildingPiece;
        const float eps = 0.001f;
        const int cols = static_cast<int>(lower.size());
        const int col = static_cast<int>(c);

        // Before comes first: when neighbours disagree it wins, so chains along the row complete.
        struct Settled { const Element* piece; Offset at; };
        std::vector<Settled> settled;
        if(col > 0) settled.push_back({&row[c - 1], {-1, 0}});
        settled.push_back({&lower[c], {0, -1}});
        if(col > 0) settled.push_back({&lower[c - 1], {-1, -1}});
        if(col + 1 < cols) settled.push_back({&lower[c + 1], {1, -1}});

        // A chain forward has to end inside the row.
        auto fits = [&](const Element& piece) {
            const Element* reg = registered(piece);
            return reg && finishFrom(cells, CHAIN_FORWARD, reg) <= static_cast<float>(cols - col) + eps;
        };
        // The piece's own links aimed back at a settled neighbour have to list it.
        auto agrees = [&](const Element& piece) {
            for(const Settled& s : settled) {
                std::vector<const Element*> wants = linkOptions(piece, s.at);
                if(!wants.empty() && !lists(wants, *s.piece)) return false;
            }
            return true;
        };

        std::vector<const Element*> allowed, pool;
        bool bound = false;
        for(const Settled& s : settled) {
            std::vector<const Element*> options = linkOptions(*s.piece, {-s.at.dx, -s.at.dy});
            if(options.empty()) continue;
            if(!bound) allowed = options;
            else std::erase_if(allowed, [&](const Element* e) { return !lists(options, *e); });
            bound = true;
            if(pool.empty()) pool = options; // the first binding neighbour's say, for when they disagree
        }
        if(bound) {
            std::vector<const Element*> best;
            for(const Element* e : allowed) if(fits(*e) && agrees(*e)) best.push_back(e);
            if(best.empty()) for(const Element* e : allowed) if(fits(*e)) best.push_back(e);
            if(best.empty()) best = allowed;
            if(!best.empty()) pool = best;
        } else {
            pool = {fits(lower[c]) && agrees(lower[c]) ? &lower[c] : &fill};
        }

        Element cell = *pool[randRange(0, static_cast<int>(pool.size()) - 1)];
        cell.width = lower[c].width; // keeps the column lined up
        return cell;
    }

    // A sample wall for the GUI, `cols` slots wide and `rowCount` rows high, walked and grown like
    // a real one but with no scene involved. Bottom row first. Seeds rand the way a build does.
    std::vector<std::vector<BuildingPiece::Element>> previewRows(uint32_t cols, uint32_t rowCount, uint32_t windowCount,
                                                                 bool mirrored, uint32_t previewSeed)
    {
        using namespace BuildingPiece;
        if(walls.empty() || cols == 0 || rowCount == 0) return {};

        srand(mixSeed(previewSeed));
        Element wall = walls[randRange(0, static_cast<int>(walls.size()) - 1)];
        std::vector<Element> elements = {wall};
        for(const Element& window : pickDistinct(windows, windowCount)) elements.push_back(window);

        const ChainTable widths = buildChainTable(false);
        const ChainTable cells = buildChainTable(true);
        // Slots the size of the wall piece, so `cols` reads as a slot count.
        const float width = std::max(wall.width, 0.001f) * static_cast<float>(cols);
        std::vector<Element> seq = walkWidth(elements, wall, mirrored ? width * 0.5f : width, widths);
        std::vector<std::vector<Element>> rows = growRows(seq, wall, rowCount, cells);
        if(mirrored) {
            for(std::vector<Element>& row : rows) mirror(row, spanBlocks(row));
        }
        return rows;
    }

    // Cuts the entrance into a finished row: every slot it covers is dropped, and the part-covered
    // slots at either end of that run are closed with stretched padding, so the row still measures
    // span.x and nothing past the opening shifts.
    void placeEntrance(BuildingPiece::Element& entrance, BuildingPiece::Element& wall, std::vector<BuildingPiece::Element>& fullSeq, glm::vec2 span)
    {
        using namespace BuildingPiece;

        if(entrance.width <= 0.0f || entrance.width > span.x) return; // nowhere to cut it in

        glm::vec2 placeRange = glm::vec2(entrance.width * 0.5f, span.x - entrance.width * 0.5f);

        // Centre of the opening, on a 0.1 grid. randRange gives 0 on an empty range, so an entrance
        // as wide as the wall is centred rather than shoved to the left edge.
        float entrancePlace = placeRange.y > placeRange.x
            ? randRange(int(10.0f*placeRange.x), int(10.0f*placeRange.y)) * 0.1f
            : placeRange.x;

        const float entranceStart = entrancePlace - entrance.width * 0.5f;
        const float entranceEnd   = entrancePlace + entrance.width * 0.5f;
        const float eps = 0.001f;

        std::vector<Element> sequence;
        sequence.reserve(fullSeq.size() + 3);

        float cursor = 0.0f;
        bool placed = false;
        for(const Element& element : fullSeq) {
            const float slotStart = cursor;
            const float slotEnd = cursor + element.width;
            cursor = slotEnd;

            // Clear of the opening on either side — the slot survives as it was.
            if(slotEnd <= entranceStart + eps || slotStart >= entranceEnd - eps) {
                sequence.push_back(element);
                continue;
            }

            // Overlapping slots are ordered, so only the first can start before the opening and
            // only the last can end after it: the pads land either side of the entrance.
            if(slotStart < entranceStart - eps) {
                Element pad = wall;
                pad.width = entranceStart - slotStart;
                sequence.push_back(pad);
            }
            if(!placed) {
                sequence.push_back(entrance);
                placed = true;
            }
            if(slotEnd > entranceEnd + eps) {
                Element pad = wall;
                pad.width = slotEnd - entranceEnd;
                sequence.push_back(pad);
            }
        }
        // A row that came up short of span.x can leave the opening past its end; it still belongs
        // in the run.
        if(!placed) sequence.push_back(entrance);

        fullSeq = std::move(sequence);
    }

    // Lays pieces edge to edge over `width`.
    std::vector<BuildingPiece::Element> walkWidth(std::vector<BuildingPiece::Element>& elements, BuildingPiece::Element& fill, float width,
                                                  const ChainTable& widths) {
        using namespace BuildingPiece;

        float start = 0.0f;
        std::vector<Element> sequence = {};

        // What each element needs to lay its shortest span; just its width when unlinked.
        std::vector<float> shortest(elements.size());
        for(uint32_t i = 0; i < elements.size(); i++) shortest[i] = shortestSpan(elements[i], widths);

        // Width still owed to elements the walk hasn't used yet. Reserving it against the space
        // left is what gets one of each into the run without forcing an order on them.
        std::vector<bool> used(elements.size(), false);
        float owed = 0.0f;
        for(float need : shortest) {
            if(need > 0.0f && std::isfinite(need)) owed += need;
        }

        std::vector<uint32_t> fits, keepsRoom, unused;

        // walk the width, picking at random from whatever still leaves room for the elements not
        // placed yet, and filling the remaining space
        while(start < width) {
            const float remaining = width - start;
            fits.clear();
            keepsRoom.clear();
            unused.clear();
            for(uint32_t i = 0; i < elements.size(); i++) {
                // Zero-width pieces are never candidates — one would leave start where it is and spin here.
                // Nor are spans that can't end (infinite).
                if(shortest[i] <= 0.0f || shortest[i] > remaining) continue;
                fits.push_back(i);
                if(!used[i]) unused.push_back(i);
                // Picking an unused element pays off its own share of what is owed.
                float owedAfter = owed - (used[i] ? 0.0f : shortest[i]);
                if(remaining - shortest[i] >= owedAfter) keepsRoom.push_back(i);
            }
            // Anything that keeps room for the rest; failing that the run is too short to fit them
            // all, so take the coverage still available before falling back to whatever fits.
            const std::vector<uint32_t>& pool = !keepsRoom.empty() ? keepsRoom : (!unused.empty() ? unused : fits);
            if(pool.empty()) break; // nothing fits in what's left; the pad below closes it

            uint32_t picked = pool[randRange(0, static_cast<int>(pool.size()) - 1)];
            // Spans may grow into room nobody is owed; on a short run they stay shortest.
            const float owedAfter = owed - (used[picked] ? 0.0f : shortest[picked]);
            const float budget = (&pool == &keepsRoom) ? remaining - owedAfter : shortest[picked];
            std::vector<Element> pieces = rollSpan(elements[picked], widths, budget);
            if(pieces.empty()) break; // guard; shortest was checked above

            for(const Element& piece : pieces) {
                sequence.push_back(piece);
                start += piece.width;
            }
            if(!used[picked]) {
                used[picked] = true;
                owed -= shortest[picked];
            }
        }

        // Close the gap with the padding piece, sized to exactly what is left, so the run measures
        // `width` whatever the pieces happened to measure.
        if(width - start > 0.001f) {
            Element pad = fill;
            pad.width = width - start;
            sequence.push_back(pad);
        }
        return sequence;
    }

    // Slot counts of a row's blocks: chained neighbours share one — the piece before links forward
    // to this one, or this one links back to it — and everything else is a block of one.
    std::vector<uint32_t> spanBlocks(const std::vector<BuildingPiece::Element>& row) const {
        std::vector<uint32_t> blocks;
        for(size_t c = 0; c < row.size(); c++) {
            const bool chained = c > 0 && (lists(linkOptions(row[c - 1], {1, 0}), row[c]) || lists(linkOptions(row[c], {-1, 0}), row[c - 1]));
            if(chained) blocks.back()++;
            else blocks.push_back(1);
        }
        return blocks;
    }

    // Reflects the half about the centre line, giving a symmetric run exactly span.x wide. The
    // centre piece appears twice on purpose — the walk stops at span.x / 2, so both copies together
    // are what fills the middle. Blocks reflect whole with their pieces kept in order, so end caps
    // stay on their ends. Works on anything laid one per slot.
    template<typename T>
    void mirror(std::vector<T>& halfSequence, const std::vector<uint32_t>& blocks) {
        std::vector<T> reflected;
        reflected.reserve(halfSequence.size());
        size_t end = halfSequence.size();
        for(auto it = blocks.rbegin(); it != blocks.rend() && *it <= end; ++it) {
            const size_t begin = end - *it;
            reflected.insert(reflected.end(), halfSequence.begin() + begin, halfSequence.begin() + end);
            end = begin;
        }
        halfSequence.insert(halfSequence.end(), reflected.begin(), reflected.end());
    }

    // Spawns the sequence edge to edge along the run's own Z from `origin`, each piece's base
    // resting on origin.y, with `yaw` turning the whole run about that origin. Laid from the far
    // end back: the run's outward face is +X, and seen from there +Z runs to the viewer's left, so
    // this is what makes "after" come out on the right, the way the rule grid and preview show it.
    void placeElements(const std::vector<BuildingPiece::Element>& sequence, glm::vec3 origin, float yaw, Scene& scene) {
        float cursor = 0.0f;
        for(const BuildingPiece::Element& element : sequence) cursor += element.width;
        for(const BuildingPiece::Element& element : sequence) {
            // Stepped regardless, so a piece that can't be spawned leaves a hole instead of
            // dragging the rest of the run out of place.
            cursor -= element.width;
            placeElement(element, cursor, origin, yaw, scene);
        }
    }

    // Spawns one piece with its leading edge `along` the run from `origin` and its base on
    // origin.y. Every spawned root is recorded so the next run can clear this one.
    void placeElement(const BuildingPiece::Element& element, float along, glm::vec3 origin, float yaw, Scene& scene) {
        using namespace BuildingPiece;

        // Variants stand in for each other, so which one lands in this slot is rolled per
        // placement — that is what keeps a repeated row from reading as a repeated row.
        if(element.templateKeys.empty()) return;
        const std::string& templateKey = element.templateKeys[randRange(0, static_cast<int>(element.templateKeys.size()) - 1)];

        auto it = scene.templates.find(templateKey);
        if(it == scene.templates.end()) return; // template dropped since it was registered
        const NodeTemplate& tmpl = it->second;

        // Stretch is the slot's width over the piece's registered width: 1 unless the walk
        // stretched a pad or a column held a piece to its width. The registered width rather than
        // the bounds, so trim that sticks out past the slot overhangs instead of being squeezed
        // in. Along the piece's own Z, which the turn below carries round with the rest of the run.
        const Element* piece = registered(element);
        const float ownWidth = piece && piece->width > 0.0f ? piece->width : tmpl.bboxMax.z - tmpl.bboxMin.z;
        float scale = ownWidth > 0.0f ? element.width / ownWidth : 1.0f;

        // A template's pivot sits wherever it was authored, so offset by its bounds: their centre
        // on the slot's centre, so an overhang spills evenly to both sides, and their base on
        // origin.y. Only Z takes the stretch — scaling the Y offset too would drop a stretched
        // piece below its row. The offset is in the run's own frame, so it turns with the wall.
        const glm::quat turn = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f));
        const float boundsMidZ = (tmpl.bboxMin.z + tmpl.bboxMax.z) * 0.5f;
        glm::vec3 localOffset(0.0f, -tmpl.bboxMin.y, along + element.width * 0.5f - boundsMidZ * scale);
        glm::vec3 spawnPos = origin + turn * localOffset;

        uint32_t root = scene.placeTemplate(templateKey, spawnPos);
        if(root == 0) return;

        // Both carry the whole piece: the bounds they were measured from covered the subtree,
        // and syncDirtyNodes picks the change up and recurses. The turn goes on the left, so
        // it rotates the placed piece about its own origin rather than reordering its authored
        // rotation.
        Node& node = scene.sceneGraph.getNode(root);
        if(yaw != 0.0f) node.relativeRotation = turn * node.relativeRotation;
        if(scale != 1.0f) node.relativeScale.z *= scale;
        node.transformDirty = true;

        spawnedNodes.push_back(root);
    }

/* ============= Serialization stuff ================== */

    // One Element block per registered piece, its bucket written as the type name. Sizes are
    // written rather than re-measured on load: parsing sees the stream and nothing else, and a
    // piece measures the same either way as long as its templates come back unchanged.
    bool serialize(std::ofstream& ofs) override {
        using namespace BuildingPiece;

        ofs << identifier << " {" << std::endl;
        for (int t = 0; t < typeCount; t++) {
            Type type = static_cast<Type>(t);
            const std::vector<Element>* bucket = elementsFor(type);
            if (!bucket) continue; // NONE has none

            for (const Element& element : *bucket) {
                ofs << "  Element {" << std::endl;
                // The bucket says what this is, not element.type — they only disagree if a piece
                // was filed wrong, and the bucket is what every lookup goes through.
                ofs << "    Type : " << typeName(type) << std::endl;
                ofs << "    Resizeable : " << (element.resizeable ? 1 : 0) << std::endl;
                ofs << "    Width : " << element.width << std::endl;
                ofs << "    Height : " << element.height << std::endl;
                ofs << "    GroundKey : " << element.groundKey << std::endl;
                // Position;Type;Key, one line per link.
                for (const LinkedElement& link : element.linkedElements) {
                    ofs << "    Link : " << linkPosName(link.linkPos) << ";" << typeName(link.type) << ";" << link.key << std::endl;
                }
                ofs << "    Templates : ";
                for (const std::string& templateKey : element.templateKeys) ofs << templateKey << ";";
                ofs << std::endl;
                ofs << "  }" << std::endl;
            }
        }
        ofs << "}" << std::endl << std::endl;
        return true;
    }

    bool parse(std::ifstream& ifs) override {
        using namespace BuildingPiece;

        clear(); // a second block would otherwise register everything twice

        std::string line, key, value;
        while (std::getline(ifs, line)) {
            SerialText::trim(line);
            if (line == "}") break; // end of the BuildingGen block
            if (line != "Element {") continue;

            Element element{NONE, false, {}, 0.0f, 0.0f};
            Type type = NONE;
            while (std::getline(ifs, line)) {
                SerialText::trim(line);
                if (line == "}") break; // end of this Element

                if (!SerialText::parseKeyValue(line, key, value)) continue;

                if (key == "Type") {
                    type = typeFromName(value);
                } else if (key == "Resizeable") {
                    element.resizeable = (std::stoi(value) != 0);
                } else if (key == "Width") {
                    element.width = std::stof(value);
                } else if (key == "Height") {
                    element.height = std::stof(value);
                } else if (key == "GroundKey") {
                    element.groundKey = value;
                } else if (key == "Link") {
                    // Unreadable links are dropped rather than guessed at.
                    const std::vector<std::string> fields = SerialText::split(value, ';');
                    LinkedPos pos = LinkedPos::SIDE_BOTH;
                    if (fields.size() == 3 && linkPosFromName(fields[0], pos)) {
                        LinkedElement link{typeFromName(fields[1]), fields[2], pos};
                        if (link.type != NONE && !link.key.empty()) element.linkedElements.push_back(link);
                    }
                } else if (key == "Templates") {
                    // Trailing separator, so split hands back an empty last field — and a template
                    // deleted since the save leaves nothing to place either.
                    for (const std::string& templateKey : SerialText::split(value, ';')) {
                        if (!templateKey.empty()) element.templateKeys.push_back(templateKey);
                    }
                }
            }

            // Without a bucket or a variant to place there is no piece here, only a slot the walk
            // would reserve width for and then leave as a hole.
            std::vector<Element>* bucket = elementsFor(type);
            if (!bucket || element.templateKeys.empty()) continue;

            element.type = type;
            bucket->push_back(element);
        }
        return true;
    }

    // Registered pieces go with the scene that defined their templates. spawnedNodes goes too:
    // those indices are handed back to the graph on a clear, so a surviving list would have the
    // next run delete nodes it never spawned.
    void clear() override {
        walls.clear();
        floorWalls.clear();
        windows.clear();
        floorWindows.clear();
        entrances.clear();
        roofs.clear();
        roofWindows.clear();
        linked.clear();
        spawnedNodes.clear();
    }
};
