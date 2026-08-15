#include "texture_converter.hpp"

#include <ktx.h>
#include <stb_image.h>
#include <stb_image_resize2.h>
#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

namespace textureconv {
namespace {

/*
Encoder tuning. These only affect how long a cold import takes and how big the cache files are —
the transcoded BC7 the GPU sees is the same shape either way, so they are safe to change (delete the
stale .ktx2 files afterwards to force a re-encode).

  UASTC_LEVEL  1 (~46.5dB) is roughly 2-3x faster than 2 (~47.5dB). 2 is the default.
  UASTC_RDO    Rate-distortion optimisation. Roughly doubles encode time and slightly lowers
               quality, in exchange for a much smaller zstd'd file. Off: the cache is a local build
               artifact, so encode time is worth more than disk here. Turn it on if the .ktx2 files
               are going into version control.
  ZSTD_LEVEL   Only affects the cache size and encode time; inflate cost at load is unaffected.
*/
constexpr ktx_pack_uastc_flags UASTC_LEVEL = KTX_PACK_UASTC_LEVEL_DEFAULT;
constexpr bool UASTC_RDO = false;
constexpr ktx_uint32_t ZSTD_LEVEL = 10;

std::mutex g_logMutex;

void logLine(const std::string& message) {
    std::lock_guard<std::mutex> lock(g_logMutex);
    std::cout << message << std::endl;
}

std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

uint32_t fullMipCount(uint32_t width, uint32_t height) {
    return static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1;
}

unsigned coreCount() {
    unsigned cores = std::thread::hardware_concurrency();
    return cores == 0 ? 1u : cores;
}

std::string ktxError(const std::string& what, KTX_error_code code) {
    return what + ": " + ktxErrorString(code);
}

bool isSourceNewer(const fs::path& source, const fs::path& cache) {
    std::error_code ec;
    auto sourceTime = fs::last_write_time(source, ec);
    if (ec)
        return false; // can't stat the source — assume the cache is fine rather than re-encoding every run
    auto cacheTime = fs::last_write_time(cache, ec);
    if (ec)
        return true;
    return sourceTime > cacheTime;
}

// Box-filtered RGBA8 mip chain. sRGB sources are downsampled in linear light — averaging
// gamma-encoded texels darkens every mip.
std::vector<std::vector<unsigned char>> buildMipChain(const unsigned char* base, uint32_t width, uint32_t height, uint32_t levels, bool srgb) {
    std::vector<std::vector<unsigned char>> chain(levels);
    chain[0].assign(base, base + static_cast<size_t>(width) * height * 4);

    uint32_t prevWidth = width;
    uint32_t prevHeight = height;
    for (uint32_t level = 1; level < levels; level++) {
        uint32_t mipWidth = std::max(1u, width >> level);
        uint32_t mipHeight = std::max(1u, height >> level);
        chain[level].resize(static_cast<size_t>(mipWidth) * mipHeight * 4);

        unsigned char* result =
            srgb ? stbir_resize_uint8_srgb(chain[level - 1].data(), static_cast<int>(prevWidth), static_cast<int>(prevHeight), 0, chain[level].data(),
                                           static_cast<int>(mipWidth), static_cast<int>(mipHeight), 0, STBIR_RGBA)
                 : stbir_resize_uint8_linear(chain[level - 1].data(), static_cast<int>(prevWidth), static_cast<int>(prevHeight), 0, chain[level].data(),
                                             static_cast<int>(mipWidth), static_cast<int>(mipHeight), 0, STBIR_RGBA);
        if (!result)
            throw std::runtime_error("mip resize failed at level " + std::to_string(level));

        prevWidth = mipWidth;
        prevHeight = mipHeight;
    }
    return chain;
}

struct KtxTextureGuard {
    ktxTexture2* tex = nullptr;
    KtxTextureGuard() = default;
    KtxTextureGuard(const KtxTextureGuard&) = delete;
    KtxTextureGuard& operator=(const KtxTextureGuard&) = delete;
    ~KtxTextureGuard() {
        if (tex)
            ktxTexture_Destroy(ktxTexture(tex));
    }
};

ColorSpace resolve(ColorSpace colorSpace) { return colorSpace == ColorSpace::Auto ? ColorSpace::Srgb : colorSpace; }

// Decode -> mip chain -> UASTC -> zstd -> .ktx2. `sources` is one image, or 6 cubemap faces.
// encodeThreads is 1 when called from the batch pool, which is already saturating the cores.
void encode(const std::vector<std::string>& sources, const std::string& outPath, ColorSpace requestedColorSpace, uint32_t encodeThreads) {
    const ColorSpace colorSpace = resolve(requestedColorSpace);
    const bool srgb = (colorSpace == ColorSpace::Srgb);
    const bool isCubemap = (sources.size() == 6);

    // Decode everything up front so a bad face fails before any encoding work is spent.
    std::vector<SourceImage> faces;
    faces.reserve(sources.size());
    for (const std::string& source : sources) {
        faces.push_back(loadSourceImage(source));
        if (faces.back().width != faces.front().width || faces.back().height != faces.front().height)
            throw std::runtime_error("cubemap faces differ in size: " + source);
    }
    if (isCubemap && faces.front().width != faces.front().height)
        throw std::runtime_error("cubemap faces must be square: " + sources.front());

    const uint32_t width = faces.front().width;
    const uint32_t height = faces.front().height;
    const uint32_t levels = fullMipCount(width, height);

    ktxTextureCreateInfo createInfo{};
    createInfo.vkFormat = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    createInfo.baseWidth = width;
    createInfo.baseHeight = height;
    createInfo.baseDepth = 1;
    createInfo.numDimensions = 2;
    createInfo.numLevels = levels;
    createInfo.numLayers = 1;
    createInfo.numFaces = isCubemap ? 6 : 1;
    createInfo.isArray = KTX_FALSE;
    createInfo.generateMipmaps = KTX_FALSE; // the chain is written explicitly below

    KtxTextureGuard texture;
    KTX_error_code result = ktxTexture2_Create(&createInfo, KTX_TEXTURE_CREATE_ALLOC_STORAGE, &texture.tex);
    if (result != KTX_SUCCESS)
        throw std::runtime_error(ktxError("ktxTexture2_Create failed", result));

    for (uint32_t face = 0; face < faces.size(); face++) {
        std::vector<std::vector<unsigned char>> chain = buildMipChain(faces[face].pixels.data(), width, height, levels, srgb);
        faces[face].pixels = {}; // the chain owns a copy of level 0 now
        for (uint32_t level = 0; level < levels; level++) {
            result = ktxTexture_SetImageFromMemory(ktxTexture(texture.tex), level, 0, face, chain[level].data(), chain[level].size());
            if (result != KTX_SUCCESS)
                throw std::runtime_error(ktxError("ktxTexture_SetImageFromMemory failed", result));
        }
    }

    ktxBasisParams params{};
    params.structSize = sizeof(params);
    params.uastc = KTX_TRUE;
    params.threadCount = encodeThreads;
    params.uastcFlags = UASTC_LEVEL;
    const bool isNormalMap = (colorSpace == ColorSpace::NormalMap);
    // Deliberately NOT setting params.normalMap. It looks like the right flag, but libktx reads it
    // as "repack this as a 2-channel normal map": it applies an rrrg swizzle, putting X in RGB and Y
    // in alpha and discarding Z for the shader to reconstruct. Our shaders sample .rgb directly, so
    // that silently turns every normal into (X, X, X). The only other thing the flag does is force
    // endpoint/selector RDO off, which is ETC1S-only and irrelevant to UASTC.
    params.normalMap = KTX_FALSE;
    // Still skipped for normal maps regardless of the global setting: RDO works by making blocks
    // more repetitive, which reads as faceting once the vectors are renormalised in the shader.
    params.uastcRDO = (UASTC_RDO && !isNormalMap) ? KTX_TRUE : KTX_FALSE;
    params.uastcRDOQualityScalar = 1.0f;
    params.uastcRDODictSize = 4096;
    params.uastcRDOMaxSmoothBlockErrorScale = 10.0f;
    params.uastcRDOMaxSmoothBlockStdDev = 18.0f;

    result = ktxTexture2_CompressBasisEx(texture.tex, &params);
    if (result != KTX_SUCCESS)
        throw std::runtime_error(ktxError("UASTC encode failed", result));

    result = ktxTexture2_DeflateZstd(texture.tex, ZSTD_LEVEL);
    if (result != KTX_SUCCESS)
        throw std::runtime_error(ktxError("zstd supercompression failed", result));

    // Write to a sibling temp file and rename, so an interrupted encode can't leave a half-written
    // .ktx2 that later runs would happily try to load.
    fs::path finalPath(outPath);
    fs::path tempPath = finalPath;
    tempPath += ".tmp";

    result = ktxTexture_WriteToNamedFile(ktxTexture(texture.tex), tempPath.string().c_str());
    if (result != KTX_SUCCESS)
        throw std::runtime_error(ktxError("failed to write " + tempPath.string(), result));

    std::error_code ec;
    fs::rename(tempPath, finalPath, ec);
    if (ec) {
        std::error_code ignored;
        fs::remove(tempPath, ignored);
        throw std::runtime_error("failed to move " + tempPath.string() + " into place: " + ec.message());
    }
}

void convertJob(const std::vector<std::string>& sources, ColorSpace colorSpace, bool force, uint32_t encodeThreads) {
    if (sources.empty())
        throw std::runtime_error("no source image given");
    if (isKtxPath(sources.front()))
        throw std::runtime_error(sources.front() + " is already a KTX file");

    const bool isCubemap = (sources.size() == 6);
    const std::string outPath = isCubemap ? cubemapKtxPathFor(sources.front()) : ktxPathFor(sources.front());
    if (!force && !(isCubemap ? cubemapNeedsConversion(sources) : needsConversion(sources.front())))
        return;

    // Logged before the work, not after: encoding a 2048^2 cubemap takes long enough that a silent
    // wait is indistinguishable from a hang.
    logLine("[ktx] encoding " + outPath + (isCubemap ? " (6 faces)" : "") + "...");

    auto start = std::chrono::steady_clock::now();
    encode(sources, outPath, colorSpace, encodeThreads);
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();

    std::error_code ec;
    auto sizeKb = fs::file_size(outPath, ec) / 1024;
    logLine("[ktx] " + outPath + " (" + std::to_string(sizeKb) + " KB, " + std::to_string(elapsedMs) + " ms)");
}

} // namespace

bool isKtxPath(const std::string& path) {
    const std::string extension = toLower(fs::path(path).extension().string());
    return extension == ".ktx" || extension == ".ktx2";
}

std::string ktxPathFor(const std::string& sourcePath) {
    if (isKtxPath(sourcePath))
        return sourcePath;
    return fs::path(sourcePath).replace_extension(".ktx2").string();
}

std::string cubemapKtxPathFor(const std::string& posXPath) {
    if (isKtxPath(posXPath))
        return posXPath;
    return fs::path(posXPath).replace_extension(".cube.ktx2").string();
}

bool needsConversion(const std::string& sourcePath) {
    if (isKtxPath(sourcePath))
        return false;
    const fs::path cache(ktxPathFor(sourcePath));
    if (!fs::exists(cache))
        return true;
    return isSourceNewer(sourcePath, cache);
}

bool cubemapNeedsConversion(const std::vector<std::string>& faces) {
    if (faces.empty() || isKtxPath(faces.front()))
        return false;
    const fs::path cache(cubemapKtxPathFor(faces.front()));
    if (!fs::exists(cache))
        return true;
    for (const std::string& face : faces) {
        if (isSourceNewer(face, cache))
            return true;
    }
    return false;
}

void convert(const std::string& sourcePath, ColorSpace colorSpace, bool force) {
    convertJob({sourcePath}, colorSpace, force, coreCount());
}

void convertCubemap(const std::vector<std::string>& faces, ColorSpace colorSpace, bool force) {
    if (faces.size() != 6)
        throw std::runtime_error("a cubemap needs exactly 6 faces");
    convertJob(faces, colorSpace, force, coreCount());
}

void convertBatch(const std::vector<Job>& jobs) {
    // A scene names the same texture from many materials; encode each output once.
    std::vector<Job> pending;
    std::set<std::string> queued;
    for (const Job& job : jobs) {
        if (job.sources.empty() || isKtxPath(job.sources.front()))
            continue;
        // A scene file can name a texture that no longer exists. Skip it quietly here and let the
        // per-material load report it, rather than logging an encoder failure for it twice.
        if (!std::all_of(job.sources.begin(), job.sources.end(), [](const std::string& s) { return fs::exists(s); }))
            continue;
        const bool isCubemap = (job.sources.size() == 6);
        if (!(isCubemap ? cubemapNeedsConversion(job.sources) : needsConversion(job.sources.front())))
            continue;
        const std::string outPath = isCubemap ? cubemapKtxPathFor(job.sources.front()) : ktxPathFor(job.sources.front());
        if (queued.insert(outPath).second)
            pending.push_back(job);
    }
    if (pending.empty())
        return;

    const unsigned workers = std::min<unsigned>(coreCount(), static_cast<unsigned>(pending.size()));
    logLine("[ktx] encoding " + std::to_string(pending.size()) + " texture(s) on " + std::to_string(workers) + " thread(s)");
    auto start = std::chrono::steady_clock::now();

    std::atomic<size_t> nextJob{0};
    std::vector<std::thread> pool;
    pool.reserve(workers);
    for (unsigned worker = 0; worker < workers; worker++) {
        pool.emplace_back([&pending, &nextJob] {
            for (size_t index = nextJob++; index < pending.size(); index = nextJob++) {
                try {
                    // One encoder thread per job: basisu's own threading would oversubscribe the pool.
                    convertJob(pending[index].sources, pending[index].colorSpace, false, 1);
                } catch (const std::exception& e) {
                    logLine(std::string("[ktx] conversion failed: ") + e.what());
                }
            }
        });
    }
    for (std::thread& thread : pool)
        thread.join();

    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    logLine("[ktx] encoded " + std::to_string(pending.size()) + " texture(s) in " + std::to_string(elapsedMs) + " ms");
}

SourceImage loadSourceImage(const std::string& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, STBI_rgb_alpha);
    if (!data) {
        const char* reason = stbi_failure_reason();
        throw std::runtime_error("Failed to load texture: " + path + " (" + (reason ? reason : "unknown") + ")");
    }

    SourceImage image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.pixels.assign(data, data + static_cast<size_t>(width) * height * 4);
    stbi_image_free(data);
    return image;
}

} // namespace textureconv
