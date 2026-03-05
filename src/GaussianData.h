#pragma once

#include <cstdint>
#include <string>
#include <vector>

// ═════════════════════════════════════════════════════════════════════════════
//  Gaussian splat data — CPU-side representation matching the standard
//  3D Gaussian Splatting .ply format.
// ═════════════════════════════════════════════════════════════════════════════

/// Per-splat data as stored in a .ply file.
struct GaussianSplat {
    // Position
    float x, y, z;

    // Normal (often unused but present in the format)
    float nx, ny, nz;

    // Spherical harmonics — DC band (3) + higher-order rest (45)
    // Total: 48 floats.  Only DC is required; rest may be zero.
    float sh[48];       // sh[0..2] = f_dc_0/1/2, sh[3..47] = f_rest_0..44

    // Opacity (logit-space in the file, sigmoid-applied at load time)
    float opacity;

    // Log-scale (3)
    float scale[3];

    // Rotation quaternion (w, x, y, z)
    float rot[4];
};

/// Result of loading a .ply file.
struct GaussianCloud {
    std::vector<GaussianSplat> splats;
    uint32_t                   shDegree = 3;  ///< SH degree detected from the file
};

// ═════════════════════════════════════════════════════════════════════════════
//  GPU-friendly layout — single unified buffer per splat.
// ═════════════════════════════════════════════════════════════════════════════

/// Unified per-splat data for the GPU (64 bytes per splat).
struct alignas(16) GaussianUnified {
    float x, y, z;                  // center position
    float opacity;                  // sigmoid-activated opacity
    float cov3D[6];                 // upper-triangle of symmetric 3×3 covariance
                                    //   c00, c01, c02, c11, c12, c22
    float shR, shG, shB;            // SH DC band coefficients
    float _pad0, _pad1, _pad2;      // pad to 64 bytes
};

/// All GPU-ready data derived from a GaussianCloud.
struct GaussianGPUData {
    std::vector<GaussianUnified> splats;
    uint32_t                     count = 0;
};

/// Convert a CPU GaussianCloud into GPU-friendly SOA layout.
GaussianGPUData prepareGPUData(const GaussianCloud& cloud);
