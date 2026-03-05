#include "GaussianData.h"

#include <cmath>

/// Sigmoid activation — converts logit-space opacity to [0, 1].
static float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/// Build 3×3 covariance matrix from scale (after exp) and normalized quaternion.
/// Returns the 6 upper-triangle values: c00, c01, c02, c11, c12, c22.
static void buildCov3D(float sx, float sy, float sz,
                       float qw, float qx, float qy, float qz,
                       float cov[6]) {
    // Rotation matrix from quaternion
    float r00 = 1.f - 2.f*(qy*qy + qz*qz);
    float r01 = 2.f*(qx*qy - qw*qz);
    float r02 = 2.f*(qx*qz + qw*qy);
    float r10 = 2.f*(qx*qy + qw*qz);
    float r11 = 1.f - 2.f*(qx*qx + qz*qz);
    float r12 = 2.f*(qy*qz - qw*qx);
    float r20 = 2.f*(qx*qz - qw*qy);
    float r21 = 2.f*(qy*qz + qw*qx);
    float r22 = 1.f - 2.f*(qx*qx + qy*qy);

    // M = R * S  (scale each column of R by the corresponding scale)
    float m00 = r00*sx, m01 = r01*sy, m02 = r02*sz;
    float m10 = r10*sx, m11 = r11*sy, m12 = r12*sz;
    float m20 = r20*sx, m21 = r21*sy, m22 = r22*sz;

    // Σ = M * Mᵀ  (symmetric → 6 unique values)
    cov[0] = m00*m00 + m01*m01 + m02*m02; // c00
    cov[1] = m00*m10 + m01*m11 + m02*m12; // c01
    cov[2] = m00*m20 + m01*m21 + m02*m22; // c02
    cov[3] = m10*m10 + m11*m11 + m12*m12; // c11
    cov[4] = m10*m20 + m11*m21 + m12*m22; // c12
    cov[5] = m20*m20 + m21*m21 + m22*m22; // c22
}

GaussianGPUData prepareGPUData(const GaussianCloud& cloud) {
    const uint32_t n = static_cast<uint32_t>(cloud.splats.size());

    GaussianGPUData gpu;
    gpu.count = n;
    gpu.splats.resize(n);

    for (uint32_t i = 0; i < n; ++i) {
        const auto& s = cloud.splats[i];
        auto& g = gpu.splats[i];

        // Center + sigmoid(opacity)
        g.x = s.x;
        g.y = s.y;
        g.z = s.z;
        g.opacity = sigmoid(s.opacity);

        // exp(scale) and normalized quaternion → 3D covariance
        float sx = std::exp(s.scale[0]);
        float sy = std::exp(s.scale[1]);
        float sz = std::exp(s.scale[2]);

        float qw = s.rot[0], qx = s.rot[1], qy = s.rot[2], qz = s.rot[3];
        float qlen = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
        if (qlen > 0.0f) { qw /= qlen; qx /= qlen; qy /= qlen; qz /= qlen; }

        buildCov3D(sx, sy, sz, qw, qx, qy, qz, g.cov3D);

        // SH DC coefficients — stored raw, shader applies the SH basis constant
        g.shR = s.sh[0];
        g.shG = s.sh[1];
        g.shB = s.sh[2];

        g._pad0 = g._pad1 = g._pad2 = 0.0f;
    }

    return gpu;
}
