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
    float3 worldPos : TEXCOORD3;
    float3 color    : COLOR0;
    float  opacity  : TEXCOORD0;
    float3 cov3Da   : TEXCOORD1;  // cov00, cov01, cov02
    float3 cov3Db   : TEXCOORD2;  // cov11, cov12, cov22
};

VSOutput main(uint vertexID : SV_VertexID) {
    GaussianData g = vk::RawBufferLoad<GaussianData>(pc.splatsAddr + vertexID * sizeof(GaussianData));

    static const float SH_C0 = 0.2820947917738781;
    float3 rgb = saturate(float3(
        0.5 + SH_C0 * g.shR,
        0.5 + SH_C0 * g.shG,
        0.5 + SH_C0 * g.shB
    ));

    float3 worldPos = float3(g.cx, g.cy, g.cz);

    VSOutput output;
    output.position  = mul(cam.proj, mul(cam.view, float4(worldPos, 1.0)));
    output.worldPos  = worldPos;
    output.color     = rgb;
    output.opacity   = g.opacity;
    output.cov3Da    = float3(g.cov0, g.cov1, g.cov2);
    output.cov3Db    = float3(g.cov3, g.cov4, g.cov5);
    return output;
}
