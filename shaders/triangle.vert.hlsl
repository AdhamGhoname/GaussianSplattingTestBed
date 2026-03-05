struct VSInput {
    [[vk::location(0)]] float2 position : POSITION;
    [[vk::location(1)]] float3 color    : COLOR;
};

struct VSOutput {
    float4 position : SV_Position;
    float3 color    : COLOR0;
};

VSOutput main(VSInput input) {
    VSOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color    = input.color;
    return output;
}
