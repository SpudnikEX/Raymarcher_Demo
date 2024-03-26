#version 460
#extension GL_GOOGLE_include_directive : enable
//#include "hg_sdf.glsl" // Current extension doesn't support #include, still compiles

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 uv;

layout(set = 0, binding = 0) uniform MvpData {
    mat4 model;
    mat4 view;
    mat4 proj;
    float time;
} uniforms;

layout(location = 0) out vec4 frag_color; //variable that is linked to the first (and only) framebuffer at index 0. (can be any name)

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const float PI = 3.14159;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Signed Distance Functions
// https://iquilezles.org/articles/distfunctions/
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Sphere SDF
/// p: point of evaluation in 3D space     point in 3D space near the sphere
/// c: point in 3D space of the origin of the sphere
/// r: radius of the sphere
float distance_from_sphere(in vec3 p, in vec3 c, float r)
{
    // return the length from the ray point, minus the radius of the circle
    // by default, the sphere is placed at 0,0,0 in screen space (-1,1). C is an offset from that
    return length(p - c) - r;
}

/// Cube SDF
/// p: current point of ray in 3D space
/// b: radius of the sphere
float sdSphere(vec3 p, float r) {
    return length(p) - r;
}

/// Cube SDF
/// p: point of evaluation in 3D space
/// c: center of the box
/// b: bounds of the box
float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

/// Cube Smooth SDF
float sdRoundBox( vec3 p, vec3 b, float r )
{
  vec3 q = abs(p) - b + r;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}

/// Torus SDF
/// p: point of evaluation in 3D space
float sdTorus( vec3 p, vec2 t )
{
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}

/// Plane SDF
/// p: raymarch evaluation of SDF in 3D space
float sdPlane( vec3 p, vec3 n, float h )
{
  // n must be normalized
  return dot(p,normalize(n)) + h;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SDF Primitive Combinations
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////


float opUnion( float d1, float d2 )
{
    return min(d1,d2);
}
float opSubtraction( float d1, float d2 )
{
    return max(-d1,d2);
}
float opIntersection( float d1, float d2 )
{
    return max(d1,d2);
}
float opXor(float d1, float d2 )
{
    return max(min(d1,d2),-max(d1,d2));
}

float opSmoothUnion( float d1, float d2, float k )
{
    float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

float opSmoothSubtraction( float d1, float d2, float k )
{
    float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
    return mix( d2, -d1, h ) + k*h*(1.0-h);
}

float opSmoothIntersection( float d1, float d2, float k )
{
    float h = clamp( 0.5 - 0.5*(d2-d1)/k, 0.0, 1.0 );
    return mix( d2, d1, h ) + k*h*(1.0-h);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Math
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float oscillate(float num, float freq, float amp) {
    return sin(uniforms.time * freq * num) * amp;
}

/// Wobble
/// https://www.ronja-tutorials.com/post/036-sdf-space-manipulation/#wobbly-space
vec3 wobble(vec3 position, float frequency, float amplitude){
    vec3 wobble = sin(position.zyx * frequency * uniforms.time) * amplitude;
    return wobble;
}

/// Wobble 3 Axis
vec3 wobble3(vec3 position, vec3 frequency, vec3 ampltude){
    vec3 wobble;
    wobble.x = sin(position.x * frequency.x * uniforms.time) * ampltude.x;
    wobble.y = sin(position.y * frequency.y * uniforms.time) * ampltude.y;
    wobble.z = sin(position.z * frequency.z * uniforms.time) * ampltude.z;
    return wobble;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Lighting
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// p: point in 3D space (the current point of the raymarch evaluation step)
float map_the_world(in vec3 p)
{
    // vec3 sphere_0_pos = vec3(-.75, 0.0, 0.0);
    // float warp = .2 + .05 * (.5 + .5 * cos (25.* (p.x+p.y) + 8.*uniforms.time));
    //float warp = cos (1.0 * (p.x+p.y) + uniforms.time) * 2.0;
    //float sphere_0 = distance_from_sphere(p, sphere_0_pos, 1.0 + warp);

    // float sphere_0 = distance_from_sphere(p, sphere_0_pos, 1.0);

    // Blob Example. Add at the end
    //    float blob = sin((p.x * p.y * p.z) * 10. - uniforms.time * 5.) * 0.025;
    float blob = sin((p.x * p.y * p.z) * 10. - uniforms.time * 5.) * 0.025;



    // Jiggle example. subtract from 
    float jiggle = sin(((p.x) + uniforms.time) * 2.0) * 0.5;
    float sphere_0 = sdSphere(p, 1.0);
    float cube_0 = sdRoundBox(p, vec3(1.0), 0.5);
    vec3 offset = vec3(-3.0, jiggle + 1.0, 1.0);


    // Inigo Displacement
    // float opDisplace( in sdf3d primitive, in vec3 p )
    // {
    //     float d1 = primitive(p);
    //     float d2 = displacement(p);
    //     return d1+d2;
    // }
    float displace = sin(20*p.x)*sin(20*p.y)*sin(20*p.z);

    //return minimum + blob;


    float sphere_1 = distance_from_sphere(p, vec3(0.0,0.0,3.0), 1.0);
    return cube_0;

    return sphere_1;// + jiggle;
    //return cube_0;
}

/// calculate normals
/// p: point in 3D space to calculate lighting from
vec3 calculate_normal(in vec3 p)
{
    const vec3 small_step = vec3(0.001, 0.0, 0.0);

    float gradient_x = map_the_world(p + small_step.xyy) - map_the_world(p - small_step.xyy);
    float gradient_y = map_the_world(p + small_step.yxy) - map_the_world(p - small_step.yxy);
    float gradient_z = map_the_world(p + small_step.yyx) - map_the_world(p - small_step.yyx);

    vec3 normal = vec3(gradient_x, gradient_y, gradient_z);

    return normalize(normal);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Raymarching Loop
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// ro: Ray Origin
/// rd: Ray Direction
vec3 ray_march(in vec3 ro, in vec3 rd)
{
    float total_distance_traveled = 0.0;
    const int NUMBER_OF_STEPS = 32; // accuracy
    const float MINIMUM_HIT_DISTANCE = 0.001;
    const float MAXIMUM_TRACE_DISTANCE = 1000.0;

    for (int i = 0; i < NUMBER_OF_STEPS; ++i)
    {
        // Point = Origin + (Length * Direction (Normalized))
        vec3 current_march_position = ro + total_distance_traveled * rd;

        // Evaluate Signed Distance Field at the current marched position for this ray
        float distance_to_closest = map_the_world(current_march_position);//map_the_world(current_march_position);

        if (distance_to_closest < MINIMUM_HIT_DISTANCE) 
        {
            // Inside a surface
            vec3 normal = calculate_normal(current_march_position);
            vec3 light_position = vec3(2.0, 5.0, 3.0);
            vec3 direction_to_light = normalize(current_march_position - light_position);

            float diffuse_intensity = max(0.0, dot(normal, direction_to_light));

            return vec3(1.0, 0.0, 0.0) * diffuse_intensity;
        }

        if (total_distance_traveled > MAXIMUM_TRACE_DISTANCE)
        {
            // Missed or outside a surface
            break;
        }
        total_distance_traveled += distance_to_closest;
    }

    //return vec3(1.0); //black
    // Gradient
    // vec3 color = mix(bg(ro, rd) * 1.5, vec3(1), 0.125);
    // color = mix(color, vec3(1), 0.5);
    // color -= dot(uv, uv * 0.155) * vec3(0.5, 1, 0.7) * 0.9;

    // Rings
    float ring = 1 - (MAXIMUM_TRACE_DISTANCE/total_distance_traveled);

    //Solid Color
    vec3 color = vec3(0.0, 0.25, 0.75) - (uv.y*.25); // y is positive downwards
    return color;

    // black = no hit, white = hit (1-distance)
    // Gives an outline effect
    //return vec3(1.0 - total_distance_traveled/MAXIMUM_TRACE_DISTANCE); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// From https://www.shadertoy.com/view/Xds3zN
// https://youtu.be/rvJHkYnAR3w?t=341
mat3 camera_to_world(vec3 ray_origin, vec3 ray_target, float camera_rotation )
{
	vec3 cz = normalize(ray_target - ray_origin); //camera_forward
	vec3 up = vec3(sin(camera_rotation), cos(camera_rotation),0.0); //Up vector for cross
	vec3 cx = normalize( cross(cz,up) ); //camera_right
	vec3 cy = normalize( cross(cx,cz) ); //camera_up
    return mat3( cx, cy, cz );
}

void main() {
    float t = uniforms.time;     // Need to process at least 1 passed uniform

    // Camera
    vec3 camera_position = vec3(sin(.5 * uniforms.time) * 5, cos(.5 * uniforms.time) * 5, -5.0);
    vec3 ray_origin = camera_position;
    mat3 view_matrix = camera_to_world(ray_origin, vec3(0),0);

    // ray direction
    // Using Screen UVs for the direction (in screen space)
    //vec3 ray_dir = normalize(vec3(uv, 1.0)); 
    float fov = 1.0;
    vec3 ray_dir = view_matrix * normalize(vec3(uv,fov));


    // Raymarch & Color
    vec3 shaded_color = ray_march(ray_origin, ray_dir);
    frag_color = vec4(shaded_color, 1.0);
    //frag_color =  vec4(uv.x, uv.y,0.0,0.0); // show uv (0-1) as color
}


/*
void main() {
    float t = uniforms.time;     // Need to process at least 1 passed uniform

    // Camera
    vec3 camera_position = vec3(0.0, 0.0, -5.0);
    vec3 ray_origin = camera_position;
    
    // ray direction
    // Using Screen UVs for the direction (in screen space)
    vec3 ray_dir = normalize(vec3(uv, 1.0)); 

    // Raymarch & Color
    vec3 shaded_color = ray_march(ray_origin, ray_dir);
    frag_color = vec4(shaded_color, 1.0);
    //frag_color =  vec4(uv.x, uv.y,0.0,0.0); // show uv (0-1) as color
}
*/
