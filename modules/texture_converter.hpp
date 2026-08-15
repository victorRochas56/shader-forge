#pragma once

#include <cstdint>
#include <string>
#include <vector>

/*
KTX2 texture cache.

Textures are authored as png/jpg but the renderer wants block-compressed data with a baked mip
chain. The first time a source image is used it is encoded to a UASTC + zstd .ktx2 sitting next to
it; every later run loads that file and transcodes it to BC7 on the way to the GPU. The .ktx2
carries the full mip chain, so nothing is blitted at runtime for file-loaded textures.

Nothing here touches Vulkan — resources.hpp owns the upload side.
*/
namespace textureconv {

// Picks the encoder's transfer function and codec tuning. Auto derives Srgb/Linear from the
// vk::Format the caller asked for; NormalMap has to be passed explicitly, because a normal map is
// linear but also needs RDO off (rate-distortion smears tangent-space vectors into visible faceting).
enum class ColorSpace { Auto, Srgb, Linear, NormalMap };

// One unit of conversion work: a single image, or the 6 faces of a cubemap in +X -X +Y -Y +Z -Z order.
struct Job {
    std::vector<std::string> sources;
    ColorSpace colorSpace = ColorSpace::Auto;
};

bool isKtxPath(const std::string& path);

// Cache path for a source image: same directory, extension swapped for .ktx2. Cubemaps get a
// .cube.ktx2 next to their +X face, so that face can still be used as a standalone 2D texture.
std::string ktxPathFor(const std::string& sourcePath);
std::string cubemapKtxPathFor(const std::string& posXPath);

// True when the cache is missing or older than its source. A cache that exists but fails to open is
// not detected here — the loader catches that and re-converts with force.
bool needsConversion(const std::string& sourcePath);
bool cubemapNeedsConversion(const std::vector<std::string>& faces);

// Encode to the .ktx2 cache. No-op when the cache is already current and force is false.
// Throws std::runtime_error on failure. Uses every core for the encode.
void convert(const std::string& sourcePath, ColorSpace colorSpace, bool force = false);
void convertCubemap(const std::vector<std::string>& faces, ColorSpace colorSpace, bool force = false);

// Converts every out-of-date job across a thread pool, one core per job, so a cold import costs
// roughly (textures / cores) encodes instead of one after another. Failures are logged and skipped:
// the per-texture load path hits them again individually and reports them in context.
void convertBatch(const std::vector<Job>& jobs);

// RGBA8 decode of a source image. Used by the encoder, and by the loader as the last-resort
// fallback when the KTX path is unusable. Throws std::runtime_error if the file can't be decoded.
struct SourceImage {
    std::vector<unsigned char> pixels; // tightly packed RGBA8
    uint32_t width = 0;
    uint32_t height = 0;
};
SourceImage loadSourceImage(const std::string& path);

} // namespace textureconv
