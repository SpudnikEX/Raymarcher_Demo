// This shader is for learning
#version 460

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color; // Dont nessecarily need this here, could be used for weight blending at some point. Colors could be input later straight into the fragment shader from a vector, to change on the fly

// Might need to change this to uniforms later
// MVP Matrices
// Time
// Light source
// Etc
layout(set = 0, binding = 0) uniform MvpData {
    mat4 model;
    mat4 view;
    mat4 proj;
    float time;
} uniforms;

layout(location=0) out vec4 out_color;
layout(location=1) out vec2 uv;

// Hardcoded fullscreen vertex positions
vec2 positions[3] = vec2[](
    vec2(-.98, -0.98),
    vec2(-.98, 3.0),
    vec2(3.0, -.98)
);

void main() {
    //gl_Position = vec4(position, 1.0); 
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0); //ignore error, still compiles
    
    out_color = color;

    // Uv's are mapped in the range of 0-1, not (-1,1)
    // For ray-casting, keep -1,1 screen coordinates (clip-space)
    uv = (gl_Position.xy / gl_Position.w);// * 0.5 + vec2(0.5, 0.5); // Maps UV to 0-1 on the screen.
    // Being a vectoral renderer, by default, vulkan uses -1,1 coordinates, so that 0,0 is the center of the window
    // -1,-1 ------- +1,-1
    //    |     |      |
    //    |    0,0     |
    //    |     |      |
    // -1,+1 ------- +1,+1
}