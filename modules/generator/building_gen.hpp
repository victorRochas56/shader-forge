#pragma once
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
        "<none>"};
    inline constexpr int typeCount = static_cast<int>(NONE) + 1;
    inline const char* typeName(Type type) { return typeNames[static_cast<int>(type)]; }

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
    };

    // The type whose elements stand in for `type` on the ground floor, or NONE if it has no
    // counterpart (the ground-floor types themselves included).
    inline Type groundFloorType(Type type) {
        switch (type) {
        case WALL:   return FLOORWALL;
        case WINDOW: return FLOORWINDOW;
        default:     return NONE;
        }
    }

}

class BuildingGen : ISerializable
{

    std::vector<BuildingPiece::Element> walls;
    std::vector<BuildingPiece::Element> floorWalls;
    std::vector<BuildingPiece::Element> windows;
    std::vector<BuildingPiece::Element> floorWindows;
    std::vector<BuildingPiece::Element> entrances;
    std::vector<BuildingPiece::Element> roofs;
    std::vector<BuildingPiece::Element> roofWindows;

    std::vector<uint32_t> spawnedNodes;
    
 public:
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

            // Pieces run along Z and stack along Y, the convention walkHalfWidth reads back.
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
        // Carry the ground-floor pairing over from the element being replaced — re-measuring a piece
        // shouldn't silently unpair it.
        for (const BuildingPiece::Element& existing : *bucket) {
            if (sharesAKey(existing)) {
                element.groundKey = existing.groundKey;
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

    void makeWall(glm::vec3 position, glm::vec2 span, uint32_t windowCount, float elementWidth, float elementHeight, Scene& scene)
    {
        using namespace BuildingPiece;

        for(uint32_t node : spawnedNodes) {
            scene.sceneGraph.removeNode(node);
        }
        spawnedNodes.clear(); // the indices are free for reuse now; keeping them would kill the new run

        Element wall = walls[randRange(0, walls.size() - 1)];
        Element floorWall = floorWalls[randRange(0, floorWalls.size() - 1)];
        Element entrance = entrances[randRange(0, entrances.size() - 1)];
        Element window_floor = floorWindows[randRange(0, floorWindows.size() - 1)];

        // The row draws from several distinct windows rather than one, so a wide wall isn't the same
        // opening repeated. walkHalfWidth reserves room for each, so all of them show up in the run.
        std::vector<Element> elements = {wall};
        for(const Element& window : pickDistinct(windows, windowCount)) elements.push_back(window);

        std::vector<Element> halfSeq = walkHalfWidth(elements, wall, span);
        std::vector<Element> halfSeq_floor;
        for(Element& el : halfSeq) {
            // Each piece hands its slot to the ground-floor element it was paired with, so a column
            // reads as the same window all the way down. Unpaired pieces fall back to the random
            // pick from the bucket, which is all this did before pairings existed.
            const Element* paired = findElement(groundFloorType(el.type), el.groundKey);
            Element floorEl = paired ? *paired : (el.type == WINDOW ? window_floor : floorWall);
            // Widths come from the row above so the columns line up — the stretched pad included,
            // which has no width of its own to keep.
            floorEl.width = el.width;
            halfSeq_floor.push_back(floorEl);
        }
        mirror(halfSeq_floor);
        mirror(halfSeq);
        //placeEntrance(entrance, wall);

        // Stack the row until the wall reaches span.y — the top row overshoots rather than leaving
        // the wall short. Row height is the tallest piece in it, so a mixed-height row still clears.
        float rowHeight = 0.0f;
        for(const Element& element : halfSeq) rowHeight = std::max(rowHeight, element.height);
        if(rowHeight <= 0.0f) return; // unmeasured pieces would never advance the stack

        placeElements(halfSeq_floor, position, scene);
        for(float y = rowHeight; y < span.y; y += rowHeight) {
            placeElements(halfSeq, position + glm::vec3(0.0f, y, 0.0f), scene);
        }
    }

    // returns the half width sequence of placed blocks
    std::vector<BuildingPiece::Element> walkHalfWidth(std::vector<BuildingPiece::Element> elements, BuildingPiece::Element fill, glm::vec2 span) {
        using namespace BuildingPiece;

        float start = 0.0f;
        std::vector<Element> sequence = {};
        const float half = span.x / 2;

        // Width still owed to elements the walk hasn't used yet. Reserving it against the space
        // left is what gets one of each into the run without forcing an order on them.
        std::vector<bool> used(elements.size(), false);
        float owed = 0.0f;
        for(const Element& element : elements) {
            if(element.width > 0.0f) owed += element.width;
        }

        std::vector<uint32_t> fits, keepsRoom, unused;

        // walk the width, picking at random from whatever still leaves room for the elements not
        // placed yet, and filling the remaining space
        while(start < half) {
            const float remaining = half - start;
            fits.clear();
            keepsRoom.clear();
            unused.clear();
            for(uint32_t i = 0; i < elements.size(); i++) {
                const Element& element = elements[i];
                // Zero-width pieces are never candidates — one would leave start where it is and spin here.
                if(element.width <= 0.0f || element.width > remaining) continue;
                fits.push_back(i);
                if(!used[i]) unused.push_back(i);
                // Picking an unused element pays off its own share of what is owed.
                float owedAfter = owed - (used[i] ? 0.0f : element.width);
                if(remaining - element.width >= owedAfter) keepsRoom.push_back(i);
            }
            // Anything that keeps room for the rest; failing that the run is too short to fit them
            // all, so take the coverage still available before falling back to whatever fits.
            const std::vector<uint32_t>& pool = !keepsRoom.empty() ? keepsRoom : (!unused.empty() ? unused : fits);
            if(pool.empty()) break; // nothing fits in what's left; the pad below closes it

            uint32_t picked = pool[randRange(0, static_cast<int>(pool.size()) - 1)];
            const Element& element = elements[picked];
            sequence.push_back(element);
            start += element.width;
            if(!used[picked]) {
                used[picked] = true;
                owed -= element.width;
            }
        }

        // Close the centre gap with the padding piece, sized to exactly what is left, so the two
        // halves still meet at span.x / 2 whatever the pieces happened to measure.
        if(half - start > 0.001f) {
            Element pad = fill;
            pad.width = half - start; 
            sequence.push_back(pad);
        }
        return sequence;
    }

    // Reflects the half about the centre line, giving a symmetric run exactly span.x wide. The
    // centre piece appears twice on purpose — the walk stops at span.x / 2, so both copies together
    // are what fills the middle. Placement still has to flip the second half about the centre;
    // the sequence only says which piece sits where.
    void mirror(std::vector<BuildingPiece::Element>& halfSequence) {
        std::vector<BuildingPiece::Element> reflected(halfSequence.rbegin(), halfSequence.rend());
        halfSequence.insert(halfSequence.end(), reflected.begin(), reflected.end());
    }

    // Spawns the sequence edge to edge along +Z from `position`, each piece's base resting on
    // position.y. Every spawned root is recorded so the next run can clear this one.
    void placeElements(const std::vector<BuildingPiece::Element>& sequence, glm::vec3 position, Scene& scene) {
        using namespace BuildingPiece;

        float cursor = 0.0f;
        for(const Element& element : sequence) {
            const float slotStart = cursor;
            // Advanced up front so a piece that can't be spawned leaves a hole instead of dragging
            // the rest of the run out of place.
            cursor += element.width;

            // Variants stand in for each other, so which one lands in this slot is rolled per
            // placement — that is what keeps a repeated row from reading as a repeated row.
            if(element.templateKeys.empty()) continue;
            const std::string& templateKey = element.templateKeys[randRange(0, static_cast<int>(element.templateKeys.size()) - 1)];

            auto it = scene.templates.find(templateKey);
            if(it == scene.templates.end()) continue; // template dropped since it was registered
            const NodeTemplate& tmpl = it->second;

            // The walk only rewrites width on the piece it stretches to close the centre gap, so
            // this is 1 for every other piece. Stretching is along the piece's own Z, which is the
            // run direction only while the template is unrotated.
            float templateWidth = tmpl.bboxMax.z - tmpl.bboxMin.z;
            float scale = templateWidth > 0.0f ? element.width / templateWidth : 1.0f;

            // A template's pivot sits wherever it was authored, so offset by its bounds to land the
            // piece's leading edge on the cursor and its base on position.y. Only Z takes the
            // stretch — scaling the Y offset too would drop a stretched piece below its row.
            glm::vec3 spawnPos = position;
            spawnPos.z += slotStart - tmpl.bboxMin.z * scale;
            spawnPos.y -= tmpl.bboxMin.y;

            uint32_t root = scene.placeTemplate(templateKey, spawnPos);
            if(root == 0) continue;

            if(scale != 1.0f) {
                // Scaling the root carries the whole piece — the bounds it was measured from
                // covered the subtree. syncDirtyNodes picks the change up and recurses.
                Node& node = scene.sceneGraph.getNode(root);
                node.relativeScale.z *= scale;
                node.transformDirty = true;
            }
            spawnedNodes.push_back(root);
        }
    }

/* ============= Serialization stuff ================== */
    const std::string identifier = "BuildingGen";

    bool serialize(std::ofstream& ofs) override {
        return true;
    }

    bool parse(std::ifstream& ifs) override {
        return true;
    }
};