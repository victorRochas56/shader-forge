#include "Simplify.h"
#include "resources.hpp"

#include <unordered_map>

// Appends progressively simplified index sets (LOD1..N) to `indices`.
// Simplification runs on position-welded topology so attribute seams (UV/normal splits)
// don't act as fake borders and crack open. Collapses only reuse existing positions, and
// each emitted corner picks the seam copy closest in attributes, so every LOD indexes the
// original vertex buffer. LODs holds the index count of each LOD: [LOD0.length, LOD1.length, ...].
void generateLODs(const std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::vector<uint32_t>& LODs) {
    constexpr int    kExtraLODs = 3;    // LOD1..LOD3
    constexpr double kKeepRatio = 0.25; // triangles kept per level
    constexpr size_t kMinTris   = 16;   // don't generate levels below this

    LODs.clear();
    LODs.push_back(static_cast<uint32_t>(indices.size()));

    const size_t triCount = indices.size() / 3;
    if (static_cast<size_t>(triCount * kKeepRatio) < kMinTris) return;

    // Weld by position: weldOf maps original vertex -> weld id, weldGroup lists the seam copies per weld
    std::unordered_map<glm::vec3, int> posToWeld;
    posToWeld.reserve(vertices.size());
    std::vector<int> weldOf(vertices.size());
    std::vector<std::vector<uint32_t>> weldGroup;
    for (size_t i = 0; i < vertices.size(); i++) {
        auto [it, inserted] = posToWeld.try_emplace(vertices[i].position, static_cast<int>(weldGroup.size()));
        if (inserted) weldGroup.emplace_back();
        weldOf[i] = it->second;
        weldGroup[it->second].push_back(static_cast<uint32_t>(i));
    }

    Simplify::vertices.clear();
    Simplify::triangles.clear();
    Simplify::refs.clear();
    Simplify::materials.clear();
    Simplify::preserve_vertices = true;

    Simplify::vertices.resize(weldGroup.size());
    for (size_t w = 0; w < weldGroup.size(); w++) {
        const glm::vec3& p = vertices[weldGroup[w][0]].position;
        Simplify::vertices[w].p = vec3f(p.x, p.y, p.z);
    }

    // material carries the source triangle id through simplification so surviving
    // corners can recover their original attributes
    std::vector<glm::uvec3> srcCorners(triCount);
    Simplify::triangles.reserve(triCount);
    for (size_t i = 0; i < triCount; i++) {
        int w0 = weldOf[indices[i * 3 + 0]];
        int w1 = weldOf[indices[i * 3 + 1]];
        int w2 = weldOf[indices[i * 3 + 2]];
        if (w0 == w1 || w1 == w2 || w2 == w0) continue; // degenerate after weld

        Simplify::Triangle t{};
        t.v[0] = w0;
        t.v[1] = w1;
        t.v[2] = w2;
        t.attr = 0;
        t.deleted = 0;
        t.dirty = 0;
        t.material = static_cast<int>(i);
        srcCorners[i] = {indices[i * 3 + 0], indices[i * 3 + 1], indices[i * 3 + 2]};
        Simplify::triangles.push_back(t);
    }

    // for a corner collapsed onto weld `w`, the original vertex there whose attributes best match `srcIdx`
    auto bestOriginal = [&](int w, uint32_t srcIdx) -> uint32_t {
        const std::vector<uint32_t>& group = weldGroup[w];
        if (group.size() == 1) return group[0];
        const Vertex& want = vertices[srcIdx];
        uint32_t best = group[0];
        float bestCost = std::numeric_limits<float>::max();
        for (uint32_t cand : group) {
            const Vertex& v = vertices[cand];
            glm::vec2 duv = v.texCoord - want.texCoord;
            float cost = glm::dot(duv, duv) + (1.0f - glm::dot(v.normal, want.normal));
            if (cost < bestCost) { bestCost = cost; best = cand; }
        }
        return best;
    };

    indices.reserve(indices.size() + indices.size() / 2);

    for (int lod = 1; lod <= kExtraLODs; lod++) {
        const size_t prevCount = Simplify::triangles.size();
        const size_t target = static_cast<size_t>(prevCount * kKeepRatio);
        if (target < kMinTris) break;

        Simplify::simplify_mesh(static_cast<int>(target));

        const size_t newCount = Simplify::triangles.size();
        if (newCount >= prevCount) break; // stalled, further levels won't shrink either

        for (const Simplify::Triangle& t : Simplify::triangles) {
            const glm::uvec3& src = srcCorners[t.material];
            for (int k = 0; k < 3; k++) {
                uint32_t o = src[k];
                indices.push_back(weldOf[o] == t.v[k] ? o : bestOriginal(t.v[k], o));
            }
        }
        LODs.push_back(static_cast<uint32_t>(newCount * 3));
    }

    Simplify::preserve_vertices = false;
}
