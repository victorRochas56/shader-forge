
struct PushConstants {
    uint one;
    float two;
    uint2 padding;    
}

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[NumThreads(64,1,1)]
[shader("compute")]
void computeMain() {

}