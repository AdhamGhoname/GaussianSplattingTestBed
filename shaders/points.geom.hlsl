struct CameraUBO {
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUBO> cam : register(b0);

struct GSInput {
    float4 position : SV_Position;
    float3 worldPos : TEXCOORD3;
    float3 color    : COLOR0;
    float  opacity  : TEXCOORD0;
    float3 cov3Da   : TEXCOORD1;
    float3 cov3Db   : TEXCOORD2;
};

struct GSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
    [[vk::builtin("PointSize")]] float pointSize : PSIZE;
};

#define MAX_SAMPLES 64

uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randFloat(uint seed) {
    return float(pcgHash(seed)) / 4294967295.0;
}

float2 boxMuller(float u1, float u2) {
    float r     = sqrt(-2.0 * log(max(u1, 1e-7)));
    float theta = 6.28318530718 * u2;
    return float2(r * cos(theta), r * sin(theta));
}

[maxvertexcount(MAX_SAMPLES + 1)]
void main(point GSInput input[1], uint primID : SV_PrimitiveID,
          inout PointStream<GSOutput> stream) {
    float3 center  = input[0].worldPos;
    float3 color   = input[0].color;
    float  opacity = input[0].opacity;

    // Upper-triangle covariance: c00, c01, c02, c11, c12, c22
    float c00 = input[0].cov3Da.x;
    float c01 = input[0].cov3Da.y;
    float c02 = input[0].cov3Da.z;
    float c11 = input[0].cov3Db.x;
    float c12 = input[0].cov3Db.y;
    float c22 = input[0].cov3Db.z;

    // Cholesky decomposition: Σ = L * Lᵀ
    float eps = 1e-6;
    float L00 = sqrt(max(c00, eps));
    float L10 = c01 / L00;
    float L20 = c02 / L00;
    float L11 = sqrt(max(c11 - L10 * L10, eps));
    float L21 = (c12 - L20 * L10) / L11;
    float L22 = sqrt(max(c22 - L20 * L20 - L21 * L21, eps));

    float viewDepth = abs(mul(cam.view, float4(center, 1.0)).z) / 1000.0f; // Normalize depth for sample count calculation
    int numSamples = clamp(int(max(opacity, viewDepth) * MAX_SAMPLES + 0.5), 0, MAX_SAMPLES);

    float4x4 viewProj = mul(cam.proj, cam.view);
    GSOutput centerOutput;
    centerOutput.position  = mul(viewProj, float4(center, 1.0));
    centerOutput.color     = color;
    centerOutput.pointSize = 1.0f;//lerp(4.0, 16.0f, clamp(viewDepth / 10.0, 0.0, 1.0)); // Adjust point size based on depth
    stream.Append(centerOutput);

    for (int i = 0; i < numSamples; i++) {
        // 4 uniform randoms -> 2 Box-Muller pairs -> 3 Gaussian values
        uint baseSeed = primID * MAX_SAMPLES + uint(i);
        float u0 = randFloat(baseSeed * 4u + 0u);
        float u1 = randFloat(baseSeed * 4u + 1u);
        float u2 = randFloat(baseSeed * 4u + 2u);
        float u3 = randFloat(baseSeed * 4u + 3u);

        float2 g01 = boxMuller(u0, u1);
        float2 g23 = boxMuller(u2, u3);
        float z0 = g01.x, z1 = g01.y, z2 = g23.x;

        // Sample offset = L * z
        float3 offset;
        offset.x = L00 * z0;
        offset.y = L10 * z0 + L11 * z1;
        offset.z = L20 * z0 + L21 * z1 + L22 * z2;

        GSOutput o;
        float3 samplePos = center + offset;
        float viewDepth = abs(mul(cam.view, float4(samplePos, 1.0)).z);
        o.position  = mul(viewProj, float4(samplePos, 1.0));
        o.color     = color;
        o.pointSize = 1.0f;//lerp(4.0, 16.0f, clamp(viewDepth / 10.0, 0.0, 1.0)); // Adjust point size based on depth
        stream.Append(o);
    }
}
