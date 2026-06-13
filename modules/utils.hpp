#pragma once
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1 // for raii
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS 1 // for structs constructors
#endif

#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <cstdio>
#include <cmath>
#include <limits>
#include <algorithm>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <filesystem>

#include "devices.hpp"

#define GLM_FORCE_RADIANS
#define GLM_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

/*
the dreaded pile of random functions
these will eventually be put into appropriate .cpp or .hpp files (TODO)
but for now here they remain
*/

static vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features, Device& device) {
    for (const auto format : candidates) {
        vk::FormatProperties props = device.getPhysicalDevice().getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        }
        if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

static uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties, Device& device) {
    vk::PhysicalDeviceMemoryProperties memProperties = device.getPhysicalDevice().getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

inline bool intersectPlane(const glm::vec3 &normal, const glm::vec3 &planeCenter, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, float& outParamDist)
{
    // Assuming vectors are all normalized
    float denom = glm::dot(normal, rayDir);
    if (std::abs(denom) > 1e-6f) {
        glm::vec3 planeRayDir = planeCenter - rayOrigin;
        outParamDist = glm::dot(planeRayDir, normal) / denom;
        return (outParamDist >= 0);
    }
    return false;
}

inline bool intersectDisk(const glm::vec3 &normal, const glm::vec3 &planeCenter, const float &radius, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir)
{
    float t = 0;
    if (intersectPlane(normal, planeCenter, rayOrigin, rayDir, t)) {
        glm::vec3 p = rayOrigin + rayDir * t; // Calculate intersection point
        glm::vec3 v = p - planeCenter; // Vector from disk center to intersection point
        float d2 = glm::dot(v, v); // Squared distance from disk center to intersection point
        return d2 <= radius * radius; // Compare squared distances (more efficient)
    }
    return false;
}

inline bool intersectCircle(const glm::vec3 &normal, const glm::vec3 &planeCenter, const float &radius, const float& thickness, const glm::vec3 &rayOrigin, const glm::vec3 &rayDir, float& outParamDist)
{
    float t = 0;
    if (intersectPlane(normal, planeCenter, rayOrigin, rayDir, t)) {
        glm::vec3 p = rayOrigin + rayDir * t; // Calculate intersection point
        glm::vec3 v = p - planeCenter; // Vector from disk center to intersection point
        float d2 = glm::dot(v, v); // Squared distance from disk center to intersection point
        bool condA = d2 <= (radius + thickness) * (radius + thickness); // Compare squared distances (more efficient)
        bool condB = d2 >= (radius - thickness) * (radius - thickness); // Compare squared distances (more efficient)
        if (condA && condB) {
            outParamDist = t;
            return true;
        }
        outParamDist = 0;
        return false;
    }
    return false;
}

inline bool intersectCylinder(/*cylinder desc*/ const glm::vec3& base, const glm::vec3& axis, const float radius, const float height,/*ray desc*/ glm::vec3& rayOrigin, glm::vec3& rayDirection, float& outParamDist) {
    // axis assumed to be a unit vector; cylinder spans [base, base + axis*height]
    glm::vec3 oc = rayOrigin - base;

    float dPar  = glm::dot(rayDirection, axis);
    float ocPar = glm::dot(oc, axis);
    glm::vec3 dPerp  = rayDirection - axis * dPar;
    glm::vec3 ocPerp = oc - axis * ocPar;

    float a = glm::dot(dPerp, dPerp);
    float b = 2.0f * glm::dot(dPerp, ocPerp);
    float c = glm::dot(ocPerp, ocPerp) - radius * radius;

    float tBest = std::numeric_limits<float>::max();
    bool  hit   = false;

    // Side surface (skip when ray is parallel to the axis -> a ~= 0)
    if (a > 1e-8f) {
        float disc = b * b - 4.0f * a * c;
        if (disc >= 0.0f) {
            float sq  = std::sqrt(disc);
            float inv = 0.5f / a;
            float t0  = (-b - sq) * inv;
            float t1  = (-b + sq) * inv;
            for (float t : {t0, t1}) {
                if (t < 0.0f || t >= tBest) continue;
                float y = ocPar + t * dPar;
                if (y >= 0.0f && y <= height) { tBest = t; hit = true; }
            }
        }
    }

    // End caps
    auto testCap = [&](const glm::vec3& center) {
        float denom = glm::dot(axis, rayDirection);
        if (std::abs(denom) < 1e-6f) return;
        float t = glm::dot(center - rayOrigin, axis) / denom;
        if (t < 0.0f || t >= tBest) return;
        glm::vec3 p = rayOrigin + rayDirection * t;
        glm::vec3 v = p - center;
        if (glm::dot(v, v) <= radius * radius) { tBest = t; hit = true; }
    };
    testCap(base);
    testCap(base + axis * height);

    if (hit) outParamDist = tBest;
    return hit;
}

static vk::Format findDepthFormat(Device& device) {
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal,
                                vk::FormatFeatureFlagBits::eDepthStencilAttachment, device);
}

static vk::SampleCountFlagBits getMaxUsableSampleCount(Device& device) {
    vk::PhysicalDeviceProperties physicalDeviceProperties = device.getPhysicalDevice().getProperties();

    vk::SampleCountFlags counts = physicalDeviceProperties.limits.framebufferColorSampleCounts & physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64) {
        return vk::SampleCountFlagBits::e64;
    }
    if (counts & vk::SampleCountFlagBits::e32) {
        return vk::SampleCountFlagBits::e32;
    }
    if (counts & vk::SampleCountFlagBits::e16) {
        return vk::SampleCountFlagBits::e16;
    }
    if (counts & vk::SampleCountFlagBits::e8) {
        return vk::SampleCountFlagBits::e8;
    }
    if (counts & vk::SampleCountFlagBits::e4) {
        return vk::SampleCountFlagBits::e4;
    }
    if (counts & vk::SampleCountFlagBits::e2) {
        return vk::SampleCountFlagBits::e2;
    }

    return vk::SampleCountFlagBits::e1;
}


static uint32_t getBytesPerPixel(vk::Format format) {
    switch (format) {
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
        return 4;
    case vk::Format::eR8G8B8Srgb:
    case vk::Format::eR8G8B8Unorm:
        return 3;
    case vk::Format::eR8G8Srgb:
    case vk::Format::eR8G8Unorm:
        return 2;
    case vk::Format::eR8Srgb:
    case vk::Format::eR8Unorm:
        return 1;
    default:
        return 4; // Default fallback
    }
}

static std::vector<char> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
}

inline bool hasFileChanged(const std::string& filePath, std::filesystem::file_time_type& lastWriteTime) {
    if(std::filesystem::last_write_time(filePath) > lastWriteTime) {
        lastWriteTime = std::filesystem::last_write_time(filePath);
        return true;
    }
    return false;
}

// Fallbacks so this still compiles if the CMake defines are absent (slangc from PATH, shaders/ next to exe).
#ifndef SLANGC_PATH
#define SLANGC_PATH "slangc"
#endif
#ifndef SHADER_SRC_DIR
#define SHADER_SRC_DIR "shaders"
#endif

// Compiles a .slang source to SPIR-V at spvOut, mirroring the CMake build flags. Captures slangc
// output and prints it on failure so shader errors surface without killing the app. Returns success.
inline bool compileSlangToSpv(const std::string& slangSrc, const std::string& spvOut) {
    namespace fs = std::filesystem;
    std::string srcDir = fs::path(slangSrc).parent_path().string();
    std::string spvAbs = fs::absolute(spvOut).string();
    std::string slangc = fs::path(SLANGC_PATH).make_preferred().string(); // backslashes for cmd.exe
    // Wrap the whole thing in an extra pair of quotes so cmd.exe tolerates spaces in the slangc path.
    std::string cmd = "\"\"" + slangc + "\" \"" + slangSrc + "\""
                      " -target spirv -profile spirv_1_4 -fvk-use-entrypoint-name -g"
                      " -I \"" + srcDir + "\" -o \"" + spvAbs + "\" 2>&1\"";

    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "[shader] failed to launch slangc for " << slangSrc << std::endl;
        return false;
    }
    std::string output;
    char buf[512];
    while (fgets(buf, sizeof(buf), pipe)) output += buf;
    int rc = _pclose(pipe);
    if (rc != 0) {
        std::cerr << "[shader] slangc failed for " << slangSrc << " (exit " << rc << "):\n" << output << std::endl;
        return false;
    }
    std::cout << "[shader] recompiled " << slangSrc << std::endl;
    return true;
}

static void decomposeTransform(const glm::mat4& matrix, glm::vec3& translation, glm::quat& rotation, glm::vec3& scale) {
    // Extract translation (4th column)
    translation = glm::vec3(matrix[3]);
    
    // Extract scale (length of first 3 columns)
    scale.x = glm::length(glm::vec3(matrix[0]));
    scale.y = glm::length(glm::vec3(matrix[1]));
    scale.z = glm::length(glm::vec3(matrix[2]));
    
    // Remove scaling from the matrix to extract rotation
    glm::mat3 rotMatrix = glm::mat3(matrix);
    rotMatrix[0] = rotMatrix[0]/ scale.x;
    rotMatrix[1] = rotMatrix[1]/ scale.y;
    rotMatrix[2] = rotMatrix[2]/ scale.z;

    // A reflection (negative determinant) can't be represented as a positive-scale
    // rotation. Fold the flip into scale.x so quat_cast receives a proper rotation matrix;
    // otherwise it returns a garbage quaternion that shears the recomposed transform.
    if (glm::determinant(rotMatrix) < 0.0f) {
        scale.x = -scale.x;
        rotMatrix[0] = -rotMatrix[0];
    }

    // Convert rotation matrix to quaternion
    rotation = glm::quat_cast(rotMatrix);
}

static glm::mat4 makeTransform( glm::vec3 translation, glm::quat rotation = glm::quat(1,0,0,0), glm::vec3 scale = glm::vec3(1.0f)) {
    glm::mat4 T = glm::translate(glm::mat4(1.0f), translation);
    glm::mat4 R = glm::mat4_cast(rotation);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);

    return T * R * S;
}

// Result of fitting a transform (rotation + optional reflection + translation) between
// two point sets. The full linear part is rotation*diag(scale); for a pure rigid fit
// scale is (1,1,1) and mirrored is false. A mirrored fit uses scale (-1,1,1).
struct RigidFit {
    glm::quat rotation;     // maps src orientation onto dst (proper rotation)
    glm::vec3 translation;  // makeTransform(translation, rotation, scale) maps src -> dst
    glm::vec3 scale;        // (1,1,1) for rigid, (-1,1,1) when a reflection was needed
    float rmsd;             // root-mean-square residual after alignment (units of the input)
    bool valid;             // false if the inputs were degenerate (too few points)
    bool mirrored;          // true when the best fit required a reflection
};

// Cyclic Jacobi eigen-decomposition for a real symmetric 4x4 matrix.
// On return d[] holds eigenvalues and the columns of v[] hold the matching eigenvectors.
// (Numerical Recipes style rotation; robust for the tiny matrices Horn's method needs.)
static void jacobiEigenSymmetric4(double a[4][4], double d[4], double v[4][4]) {
    const int n = 4;
    double b[4][4];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b[i][j] = a[i][j];
            v[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) off += std::abs(b[p][q]);
        if (off < 1e-18) break;

        for (int p = 0; p < n; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = b[p][q];
                if (std::abs(apq) < 1e-300) continue;
                double app = b[p][p], aqq = b[q][q];
                double theta = (aqq - app) / (2.0 * apq);
                double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                double c = 1.0 / std::sqrt(t * t + 1.0);
                double s = t * c;
                // Apply the rotation as J^T * B * J (columns then rows), and accumulate into v.
                for (int i = 0; i < n; i++) {
                    double bip = b[i][p], biq = b[i][q];
                    b[i][p] = c * bip - s * biq;
                    b[i][q] = s * bip + c * biq;
                }
                for (int i = 0; i < n; i++) {
                    double bpi = b[p][i], bqi = b[q][i];
                    b[p][i] = c * bpi - s * bqi;
                    b[q][i] = s * bpi + c * bqi;
                }
                for (int i = 0; i < n; i++) {
                    double vip = v[i][p], viq = v[i][q];
                    v[i][p] = c * vip - s * viq;
                    v[i][q] = s * vip + c * viq;
                }
            }
        }
    }
    for (int i = 0; i < n; i++) d[i] = b[i][i];
}

// Best-fit rigid transform (R, t) such that R*src[i] + t ~= dst[i], assuming src[i]
// corresponds to dst[i]. Uses Horn's quaternion method, so the result is always a
// proper rotation (no reflections). The returned rmsd is the alignment residual: a
// small value means the correspondence held; a large value means it did not (e.g. a
// welded format reordered the vertices) and the caller should fall back.
static RigidFit kabsch(const std::vector<glm::vec3>& src, const std::vector<glm::vec3>& dst) {
    RigidFit fit{glm::quat(1, 0, 0, 0), glm::vec3(0.0f), glm::vec3(1.0f), std::numeric_limits<float>::max(), false, false};

    const size_t n = std::min(src.size(), dst.size());
    if (n < 3) return fit;

    // Centroids (double precision to keep the covariance well-conditioned).
    glm::dvec3 cs(0.0), cd(0.0);
    for (size_t i = 0; i < n; i++) { cs += glm::dvec3(src[i]); cd += glm::dvec3(dst[i]); }
    cs /= double(n);
    cd /= double(n);

    // Cross-covariance S[j][k] = sum (src-cs)[j] * (dst-cd)[k].
    double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (size_t i = 0; i < n; i++) {
        glm::dvec3 a = glm::dvec3(src[i]) - cs;
        glm::dvec3 b = glm::dvec3(dst[i]) - cd;
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) S[j][k] += a[j] * b[k];
    }

    // Horn's symmetric 4x4 matrix; its largest-eigenvalue eigenvector is the quaternion.
    double Sxx = S[0][0], Sxy = S[0][1], Sxz = S[0][2];
    double Syx = S[1][0], Syy = S[1][1], Syz = S[1][2];
    double Szx = S[2][0], Szy = S[2][1], Szz = S[2][2];
    double N[4][4] = {
        { Sxx + Syy + Szz, Syz - Szy,        Szx - Sxz,        Sxy - Syx        },
        { Syz - Szy,       Sxx - Syy - Szz,  Sxy + Syx,        Szx + Sxz        },
        { Szx - Sxz,       Sxy + Syx,       -Sxx + Syy - Szz,  Syz + Szy        },
        { Sxy - Syx,       Szx + Sxz,        Syz + Szy,       -Sxx - Syy + Szz  },
    };

    double d[4], v[4][4];
    jacobiEigenSymmetric4(N, d, v);

    int best = 0;
    for (int i = 1; i < 4; i++) if (d[i] > d[best]) best = i;

    // Eigenvector column 'best' is the quaternion (w, x, y, z).
    glm::quat q(static_cast<float>(v[0][best]), static_cast<float>(v[1][best]),
                static_cast<float>(v[2][best]), static_cast<float>(v[3][best]));
    if (glm::dot(q, q) < 1e-12f) return fit; // degenerate
    q = glm::normalize(q);

    glm::dvec3 t = cd - glm::dvec3(glm::dmat3(glm::mat3_cast(q)) * cs);

    fit.rotation = q;
    fit.translation = glm::vec3(t);
    fit.valid = true;

    // Residual: how well the recovered transform maps src onto dst.
    double sse = 0.0;
    for (size_t i = 0; i < n; i++) {
        glm::vec3 mapped = fit.rotation * src[i] + fit.translation;
        sse += glm::dot(glm::dvec3(mapped) - glm::dvec3(dst[i]), glm::dvec3(mapped) - glm::dvec3(dst[i]));
    }
    fit.rmsd = float(std::sqrt(sse / double(n)));
    return fit;
}

// Uniform spatial grid over a point cloud for approximate nearest-neighbour queries.
// Used by ICP so each correspondence lookup is ~O(1) instead of O(N).
struct PointGrid {
    const std::vector<glm::vec3>* pts = nullptr;
    glm::vec3 origin{0.0f};
    float inv = 1.0f; // 1 / cellSize
    int nx = 1, ny = 1, nz = 1;
    std::vector<std::vector<int>> cells;

    int cellIndex(int x, int y, int z) const { return (z * ny + y) * nx + x; }

    glm::ivec3 cellOf(const glm::vec3& p) const {
        glm::vec3 r = (p - origin) * inv;
        return glm::ivec3(glm::clamp(int(r.x), 0, nx - 1),
                          glm::clamp(int(r.y), 0, ny - 1),
                          glm::clamp(int(r.z), 0, nz - 1));
    }

    void build(const std::vector<glm::vec3>& points, int targetPerCell = 4) {
        pts = &points;
        glm::vec3 mn(std::numeric_limits<float>::max());
        glm::vec3 mx(std::numeric_limits<float>::lowest());
        for (const auto& p : points) { mn = glm::min(mn, p); mx = glm::max(mx, p); }
        glm::vec3 size = glm::max(mx - mn, glm::vec3(1e-4f));
        int targetCells = std::max(1, int(points.size()) / std::max(1, targetPerCell));
        double volume = double(size.x) * size.y * size.z;
        float cell = float(std::cbrt(volume / std::max(1, targetCells)));
        if (!(cell > 0.0f)) cell = 1.0f;
        inv = 1.0f / cell;
        origin = mn;
        nx = std::max(1, int(size.x * inv) + 1);
        ny = std::max(1, int(size.y * inv) + 1);
        nz = std::max(1, int(size.z * inv) + 1);
        cells.assign(size_t(nx) * ny * nz, {});
        for (int i = 0; i < int(points.size()); i++) {
            glm::ivec3 c = cellOf(points[i]);
            cells[cellIndex(c.x, c.y, c.z)].push_back(i);
        }
    }

    // Index of the nearest stored point to q. Searches in expanding shells; once a
    // candidate is found it scans one extra shell so the result is essentially exact.
    int nearest(const glm::vec3& q) const {
        const auto& P = *pts;
        glm::ivec3 c = cellOf(q);
        int best = -1;
        float bestD = std::numeric_limits<float>::max();
        int maxR = std::max({nx, ny, nz});
        int foundAt = -1;
        for (int r = 0; r <= maxR; r++) {
            int x0 = std::max(0, c.x - r), x1 = std::min(nx - 1, c.x + r);
            int y0 = std::max(0, c.y - r), y1 = std::min(ny - 1, c.y + r);
            int z0 = std::max(0, c.z - r), z1 = std::min(nz - 1, c.z + r);
            for (int z = z0; z <= z1; z++)
                for (int y = y0; y <= y1; y++)
                    for (int x = x0; x <= x1; x++) {
                        // Only visit the current shell; inner cells were already searched.
                        if (r > 0 && x > c.x - r && x < c.x + r && y > c.y - r && y < c.y + r &&
                            z > c.z - r && z < c.z + r)
                            continue;
                        for (int pi : cells[cellIndex(x, y, z)]) {
                            glm::vec3 diff = P[pi] - q;
                            float d = glm::dot(diff, diff);
                            if (d < bestD) { bestD = d; best = pi; }
                        }
                    }
            if (best >= 0) {
                if (foundAt < 0) foundAt = r;       // first hit
                else if (r >= foundAt + 1) break;   // one extra shell, then stop
            }
        }
        return best;
    }
};

// The 24 proper-rotation orientations made of signed axis permutations (det == +1).
// These coarsely cover rotation space (any rotation is within ~60deg of one), so they
// make good multi-start seeds for ICP.
static std::vector<glm::mat3> buildOrientationSeeds() {
    std::vector<glm::mat3> seeds;
    const int perms[6][3] = {{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
    for (const auto& pm : perms) {
        for (int s = 0; s < 8; s++) {
            glm::vec3 sgn((s & 1) ? 1.0f : -1.0f, (s & 2) ? 1.0f : -1.0f, (s & 4) ? 1.0f : -1.0f);
            glm::mat3 M(0.0f);          // column j (glm is column-major) is sgn[j] * e_{pm[j]}
            M[0][pm[0]] = sgn[0];
            M[1][pm[1]] = sgn[1];
            M[2][pm[2]] = sgn[2];
            if (glm::determinant(M) > 0.0f) seeds.push_back(M);
        }
    }
    return seeds;
}

// Dominant principal axis of a point cloud (eigenvector of the largest covariance
// eigenvalue) via power iteration. For an extruded shape like a column this is the long
// axis -- exactly the axis whose spin ICP struggles to recover from coarse seeds.
static glm::vec3 dominantAxis(const std::vector<glm::vec3>& pts) {
    glm::dvec3 c(0.0);
    for (const auto& p : pts) c += glm::dvec3(p);
    c /= double(pts.size());
    double C[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (const auto& p : pts) {
        glm::dvec3 d = glm::dvec3(p) - c;
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) C[j][k] += d[j] * d[k];
    }
    glm::dvec3 v(0.5773502692, 0.5773502692, 0.5773502692); // generic start, not axis-aligned
    for (int iter = 0; iter < 40; iter++) {
        glm::dvec3 nv(C[0][0]*v.x + C[0][1]*v.y + C[0][2]*v.z,
                      C[1][0]*v.x + C[1][1]*v.y + C[1][2]*v.z,
                      C[2][0]*v.x + C[2][1]*v.y + C[2][2]*v.z);
        double len = glm::length(nv);
        if (len < 1e-12) break;
        v = nv / len;
    }
    return glm::normalize(glm::vec3(v));
}

// Shortest-arc rotation taking unit vector `from` onto unit vector `to`.
static glm::mat3 alignAxis(glm::vec3 from, glm::vec3 to) {
    from = glm::normalize(from);
    to = glm::normalize(to);
    float d = glm::dot(from, to);
    if (d > 0.9999f) return glm::mat3(1.0f);
    if (d < -0.9999f) { // antiparallel: rotate 180 about any perpendicular axis
        glm::vec3 ortho = (std::abs(from.x) < 0.9f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
        glm::vec3 axis = glm::normalize(glm::cross(from, ortho));
        return glm::mat3_cast(glm::angleAxis(3.14159265358979f, axis));
    }
    glm::vec3 axis = glm::normalize(glm::cross(from, to));
    float angle = std::acos(glm::clamp(d, -1.0f, 1.0f));
    return glm::mat3_cast(glm::angleAxis(angle, axis));
}

// Append seeds that align the src dominant axis to the dst dominant axis (both polarities)
// and then sweep K rotations about it. This targets the spin-about-axis DOF that the coarse
// orientation seeds sample too sparsely for near-axisymmetric meshes.
static void appendSpinSeeds(std::vector<glm::mat3>& seeds, glm::vec3 srcAxis, glm::vec3 dstAxis, int K) {
    const float twoPi = 6.28318530717959f;
    glm::vec3 spinAxis = glm::normalize(dstAxis);
    for (int flip = 0; flip < 2; flip++) {
        glm::vec3 target = flip ? -dstAxis : dstAxis;
        glm::mat3 R0 = alignAxis(srcAxis, target);
        for (int k = 0; k < K; k++) {
            float th = (twoPi * k) / float(K);
            glm::mat3 spin = glm::mat3_cast(glm::angleAxis(th, spinAxis));
            seeds.push_back(spin * R0);
        }
    }
}

// Iterative Closest Point restricted to proper rotations. Runs ICP from each seed
// orientation. Correspondences are by position, but the WINNER is chosen with normals
// folded in: among seeds whose positional residual is within a band of the best, the one
// whose transformed src normals best agree with the dst normals wins. This disambiguates
// position-symmetric meshes (where several orientations fit the points equally well but
// only one matches the surface detail). samp/sampN and dst/dstN are parallel arrays;
// normals are transformed by R (valid since R is orthonormal). outNormDefect is the mean
// normal mismatch (0 = perfect, up to 2) of the chosen fit.
static void properICP(const std::vector<glm::vec3>& samp, const std::vector<glm::vec3>& sampN,
                      const std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& dstN,
                      const PointGrid& grid, const std::vector<glm::mat3>& seeds,
                      const glm::vec3& dstCentroid, const glm::vec3& dstAxis,
                      int maxIters, float earlyThreshold, float diag,
                      glm::mat3& outR, glm::vec3& outT, float& outPosRmsd, float& outNormDefect) {
    outR = glm::mat3(1.0f);
    outT = glm::vec3(0.0f);
    outPosRmsd = std::numeric_limits<float>::max();
    outNormDefect = 2.0f;

    glm::dvec3 csd(0.0);
    for (const auto& p : samp) csd += glm::dvec3(p);
    csd /= double(samp.size());
    glm::vec3 cs(csd);

    // Targeted spin-about-axis seeds (tried first, so early-out can fire on them), plus the
    // coarse orientation seeds for everything else.
    std::vector<glm::mat3> allSeeds;
    allSeeds.reserve(seeds.size() + 24);
    appendSpinSeeds(allSeeds, dominantAxis(samp), dstAxis, 12);
    allSeeds.insert(allSeeds.end(), seeds.begin(), seeds.end());

    std::vector<glm::vec3> matchedSrc(samp.size());
    std::vector<glm::vec3> matchedDst(samp.size());

    struct SeedResult { glm::mat3 R; glm::vec3 t; float pos; float norm; };
    std::vector<SeedResult> results;
    results.reserve(allSeeds.size());

    for (const glm::mat3& seed : allSeeds) {
        glm::mat3 R = seed;
        glm::vec3 t = dstCentroid - R * cs; // start with centroids aligned
        float prevRmsd = std::numeric_limits<float>::max();

        for (int iter = 0; iter < maxIters; iter++) {
            double sse = 0.0;
            for (size_t i = 0; i < samp.size(); i++) {
                glm::vec3 y = R * samp[i] + t;
                int j = grid.nearest(y);
                matchedSrc[i] = samp[i];
                matchedDst[i] = dst[j];
                glm::vec3 diff = y - dst[j];
                sse += glm::dot(diff, diff);
            }
            float rmsd = float(std::sqrt(sse / double(samp.size())));

            // Re-solve the best proper transform for the current correspondences.
            RigidFit f = kabsch(matchedSrc, matchedDst);
            if (!f.valid) break;
            R = glm::mat3_cast(f.rotation);
            t = f.translation;

            if (prevRmsd - rmsd < 1e-4f * prevRmsd) break; // converged
            prevRmsd = rmsd;
        }

        // Final positional residual + normal mismatch with the converged transform.
        double sse = 0.0, ndSum = 0.0;
        int ndCount = 0;
        for (size_t i = 0; i < samp.size(); i++) {
            glm::vec3 y = R * samp[i] + t;
            int j = grid.nearest(y);
            glm::vec3 diff = y - dst[j];
            sse += glm::dot(diff, diff);

            glm::vec3 na = R * sampN[i]; // normal transform == R (orthonormal)
            float la = glm::length(na), lb = glm::length(dstN[j]);
            if (la > 1e-6f && lb > 1e-6f) {
                float d = glm::clamp(glm::dot(na / la, dstN[j] / lb), -1.0f, 1.0f);
                ndSum += 1.0f - d;
                ndCount++;
            }
        }
        float pos = float(std::sqrt(sse / double(samp.size())));
        float nd = ndCount ? float(ndSum / ndCount) : 0.0f;
        results.push_back({R, t, pos, nd});

        if (pos < earlyThreshold && nd < 0.05f) break; // fully correct (position + normals)
    }

    if (results.empty()) return;

    // Band-select: among fits that explain the points about as well as the best, prefer the
    // one with the best normal agreement.
    float minPos = std::numeric_limits<float>::max();
    for (const auto& r : results) minPos = std::min(minPos, r.pos);
    float band = minPos * 1.5f + 1e-4f * diag;

    int best = 0;
    float bestNd = std::numeric_limits<float>::max();
    for (int i = 0; i < int(results.size()); i++) {
        if (results[i].pos <= band && results[i].norm < bestNd) { bestNd = results[i].norm; best = i; }
    }

    outR = results[best].R;
    outT = results[best].t;
    outPosRmsd = results[best].pos;
    outNormDefect = results[best].norm;
}

// Best-fit transform mapping the src point set onto the dst point set WITHOUT assuming
// any vertex correspondence (welded formats reorder vertices). Uses multi-start ICP, with
// normals used to disambiguate symmetric meshes, and optionally tries a reflection so
// mirrored instances can still be matched. src/srcNormals and dst/dstNormals are parallel
// arrays. src is subsampled for speed; the residual (rmsd) reports alignment quality so
// the caller can reject bad fits.
static RigidFit icpAlign(const std::vector<glm::vec3>& src, const std::vector<glm::vec3>& srcNormals,
                         const std::vector<glm::vec3>& dst, const std::vector<glm::vec3>& dstNormals,
                         bool allowReflection = true, int maxSamples = 256, int maxIters = 25) {
    RigidFit fit{glm::quat(1, 0, 0, 0), glm::vec3(0.0f), glm::vec3(1.0f),
                 std::numeric_limits<float>::max(), false, false};
    if (src.size() < 3 || dst.size() < 3) return fit;

    // Subsample src (and its normals) to bound the cost of each ICP iteration.
    std::vector<glm::vec3> samp, sampN;
    size_t stride = std::max<size_t>(1, src.size() / std::max(1, maxSamples));
    samp.reserve(src.size() / stride + 1);
    sampN.reserve(src.size() / stride + 1);
    bool haveSrcN = srcNormals.size() == src.size();
    for (size_t i = 0; i < src.size(); i += stride) {
        samp.push_back(src[i]);
        sampN.push_back(haveSrcN ? srcNormals[i] : glm::vec3(0.0f));
    }
    if (samp.size() < 3) { samp = src; sampN = haveSrcN ? srcNormals : std::vector<glm::vec3>(src.size(), glm::vec3(0.0f)); }

    // dst normals must be index-aligned with dst; fall back to zero (neutral) if not.
    const std::vector<glm::vec3> zeroN(dst.size(), glm::vec3(0.0f));
    const std::vector<glm::vec3>& dstN = (dstNormals.size() == dst.size()) ? dstNormals : zeroN;

    PointGrid grid;
    grid.build(dst);

    glm::vec3 mn(std::numeric_limits<float>::max());
    glm::vec3 mx(std::numeric_limits<float>::lowest());
    glm::dvec3 cdd(0.0);
    for (const auto& p : dst) { mn = glm::min(mn, p); mx = glm::max(mx, p); cdd += glm::dvec3(p); }
    cdd /= double(dst.size());
    glm::vec3 dstCentroid(cdd);
    float diag = glm::length(mx - mn);
    float earlyThreshold = diag * 1e-3f; // stop seeds once near-perfect
    glm::vec3 dstAxis = dominantAxis(dst);

    std::vector<glm::mat3> seeds = buildOrientationSeeds();

    // Proper (non-reflected) alignment.
    glm::mat3 R; glm::vec3 t; float pos, nd;
    properICP(samp, sampN, dst, dstN, grid, seeds, dstCentroid, dstAxis, maxIters, earlyThreshold, diag, R, t, pos, nd);

    glm::mat3 bestL = R;        // linear part mapping original src -> dst
    glm::vec3 bestT = t;
    float bestPos = pos, bestNd = nd;
    bool mirrored = false;

    if (allowReflection) {
        // Reflect the source (positions and normals) across X about the origin, then look for
        // a proper rotation; the combined map is improper, i.e. a reflection.
        std::vector<glm::vec3> sampM(samp.size()), sampMN(samp.size());
        for (size_t i = 0; i < samp.size(); i++) {
            sampM[i]  = glm::vec3(-samp[i].x,  samp[i].y,  samp[i].z);
            sampMN[i] = glm::vec3(-sampN[i].x, sampN[i].y, sampN[i].z);
        }

        glm::mat3 R2; glm::vec3 t2; float pos2, nd2;
        properICP(sampM, sampMN, dst, dstN, grid, seeds, dstCentroid, dstAxis, maxIters, earlyThreshold, diag, R2, t2, pos2, nd2);

        // Choose proper vs mirror by a combined position+normal cost (lower is better) so a
        // reflection only wins when it genuinely fits better, not by a positional hair.
        float costProper = bestPos / std::max(diag, 1e-6f) + 0.5f * bestNd;
        float costMirror = pos2    / std::max(diag, 1e-6f) + 0.5f * nd2;
        if (costMirror < costProper) {
            glm::mat3 D(1.0f); D[0][0] = -1.0f;     // diag(-1,1,1)
            bestL = R2 * D;                          // y = R2*(D*p) + t2 for original p
            bestT = t2;
            bestPos = pos2;
            bestNd = nd2;
            mirrored = true;
        }
    }

    fit.rmsd = bestPos;
    fit.translation = bestT;
    fit.valid = true;
    fit.mirrored = mirrored;
    // scale stays (1,1,1): callers handle a reflection by using mirrored geometry + a proper
    // rotation, never a negative scale (which would shear once the node system decomposes it).
    fit.scale = glm::vec3(1.0f);
    if (!mirrored) {
        fit.rotation = glm::normalize(glm::quat_cast(bestL));
    } else {
        glm::mat3 Rp = bestL; Rp[0] = -Rp[0];        // proper part: Rp = bestL * diag(-1,1,1)
        fit.rotation = glm::normalize(glm::quat_cast(Rp));
    }
    return fit;
}

// Frustum culling structures and functions
struct Plane {
    glm::vec3 normal;
    float distance;
};

// Extracts the 6 frustum planes from a light space matrix
inline std::array<Plane, 6> extractFrustumPlanes(const glm::mat4& lightSpaceMatrix) {
    std::array<Plane, 6> planes;

    // Left plane
    planes[0].normal = glm::vec3(lightSpaceMatrix[0][3] + lightSpaceMatrix[0][0], lightSpaceMatrix[1][3] + lightSpaceMatrix[1][0], lightSpaceMatrix[2][3] + lightSpaceMatrix[2][0]);
    planes[0].distance = lightSpaceMatrix[3][3] + lightSpaceMatrix[3][0];

    // Right plane
    planes[1].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][0], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][0], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][0]);
    planes[1].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][0];

    // Bottom plane
    planes[2].normal = glm::vec3(lightSpaceMatrix[0][3] + lightSpaceMatrix[0][1], lightSpaceMatrix[1][3] + lightSpaceMatrix[1][1], lightSpaceMatrix[2][3] + lightSpaceMatrix[2][1]);
    planes[2].distance = lightSpaceMatrix[3][3] + lightSpaceMatrix[3][1];

    // Top plane
    planes[3].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][1], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][1], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][1]);
    planes[3].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][1];

    // Near plane (Vulkan/GLM_DEPTH_ZERO_TO_ONE: depth range [0,1], near at z_ndc=0, so just row2)
    planes[4].normal = glm::vec3(lightSpaceMatrix[0][2], lightSpaceMatrix[1][2], lightSpaceMatrix[2][2]);
    planes[4].distance = lightSpaceMatrix[3][2];

    // Far plane
    planes[5].normal = glm::vec3(lightSpaceMatrix[0][3] - lightSpaceMatrix[0][2], lightSpaceMatrix[1][3] - lightSpaceMatrix[1][2], lightSpaceMatrix[2][3] - lightSpaceMatrix[2][2]);
    planes[5].distance = lightSpaceMatrix[3][3] - lightSpaceMatrix[3][2];

    // Normalize all planes
    for (auto& plane : planes) {
        float length = glm::length(plane.normal);
        plane.normal /= length;
        plane.distance /= length;
    }

    return planes;
}

inline bool isAABBInFrustum(const glm::vec3& aabbMin, const glm::vec3& aabbMax, const std::array<Plane, 6>& planes, float epsilon = 0.01f) {
    // Test the AABB against each plane
    for (const auto& plane : planes) {
        glm::vec3 positiveVertex;
        positiveVertex.x = (plane.normal.x >= 0.0f) ? aabbMax.x : aabbMin.x;
        positiveVertex.y = (plane.normal.y >= 0.0f) ? aabbMax.y : aabbMin.y;
        positiveVertex.z = (plane.normal.z >= 0.0f) ? aabbMax.z : aabbMin.z;

        // Negative epsilon makes the frustum "bigger" (more conservative culling)
        if (glm::dot(plane.normal, positiveVertex) + plane.distance < -epsilon) {
            return false;
        }
    }

    return true;
}

// Helper function to transform a local-space AABB to world space
inline void transformAABBToWorldSpace(const glm::vec3& localMin, const glm::vec3& localMax,
                                      const glm::mat4& worldTransform,
                                      glm::vec3& worldMin, glm::vec3& worldMax) {
    // Transform all 8 corners and find new AABB in world space
    glm::vec3 corners[8] = {
        glm::vec3(localMin.x, localMin.y, localMin.z),
        glm::vec3(localMax.x, localMin.y, localMin.z),
        glm::vec3(localMin.x, localMax.y, localMin.z),
        glm::vec3(localMax.x, localMax.y, localMin.z),
        glm::vec3(localMin.x, localMin.y, localMax.z),
        glm::vec3(localMax.x, localMin.y, localMax.z),
        glm::vec3(localMin.x, localMax.y, localMax.z),
        glm::vec3(localMax.x, localMax.y, localMax.z)
    };

    worldMin = glm::vec3(std::numeric_limits<float>::max());
    worldMax = glm::vec3(std::numeric_limits<float>::lowest());

    for (const auto& corner : corners) {
        glm::vec3 worldCorner = glm::vec3(worldTransform * glm::vec4(corner, 1.0f));
        worldMin = glm::min(worldMin, worldCorner);
        worldMax = glm::max(worldMax, worldCorner);
    }
}