struct CameraUBO {
    float4x4 view;
    float4x4 proj;
};

[[vk::binding(0, 0)]] ConstantBuffer<CameraUBO> cam : register(b0);

[[vk::push_constant]]
struct PushConstants {
    uint64_t splatsAddr;
} pc;

struct GaussianData {
    float cx, cy, cz;
    float opacity;
    float cov0, cov1, cov2, cov3, cov4, cov5;
    float shR, shG, shB;
    float _pad0, _pad1, _pad2;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
    [[vk::builtin("PointSize")]] float pointSize : PSIZE;
};

VSOutput main(uint vertexID : SV_VertexID) {
    GaussianData g = vk::RawBufferLoad<GaussianData>(pc.splatsAddr + vertexID * sizeof(GaussianData));

    static const float SH_C0 = 0.2820947917738781;
    float3 rgb = saturate(float3(
        0.5 + SH_C0 * g.shR,
        0.5 + SH_C0 * g.shG,
        0.5 + SH_C0 * g.shB
    ));

    VSOutput output;
    float4 worldPos = float4(g.cx, g.cy, g.cz, 1.0);
    output.position  = mul(cam.proj, mul(cam.view, worldPos));
    output.color     = rgb;
    output.pointSize = 1.0;
    return output;
}
