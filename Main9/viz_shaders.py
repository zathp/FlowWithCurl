SKYBOX_VERT = r"""
#version 330 core
layout (location = 0) in vec3 aPos;
out vec3 vDir;
uniform mat4 uInvViewProj;
void main() {
    vec4 p = vec4(aPos, 1.0);
    vec4 w = uInvViewProj * p;
    vDir = w.xyz / w.w;
    gl_Position = p;
}
"""

SKYBOX_FRAG = r"""
#version 330 core
in vec3 vDir;
out vec4 FragColor;
void main() {
    vec3 d = normalize(vDir);
    vec3 c;
    if (abs(d.x) > abs(d.y) && abs(d.x) > abs(d.z))
        c = (d.x > 0.0) ? vec3(1,0,0) : vec3(0,1,0);
    else if (abs(d.y) > abs(d.z))
        c = (d.y > 0.0) ? vec3(0,0,1) : vec3(1,1,0);
    else
        c = (d.z > 0.0) ? vec3(1,0,1) : vec3(0,1,1);
    FragColor = vec4(c, 1.0);
}
"""

FS_TRI_VERT = r"""
#version 330 core
out vec2 vUV;
void main() {
    vec2 p;
    if (gl_VertexID == 0) p = vec2(-1.0, -1.0);
    else if (gl_VertexID == 1) p = vec2( 3.0, -1.0);
    else p = vec2(-1.0,  3.0);
    vUV = 0.5 * (p + 1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

VOLUME_FRAG = r"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler3D uDensityTex;
uniform mat4 uInvViewProj;
uniform vec3 uCamPos;

uniform vec3 uBoxMin;
uniform vec3 uBoxMax;

uniform float uAbsorption;
uniform int   uSteps;
uniform float uDensityScale;
uniform float uDensityBias;

bool intersectAABB(vec3 ro, vec3 rd, vec3 bmin, vec3 bmax, out vec2 tHit) {
    vec3 invD = 1.0 / rd;
    vec3 t0 = (bmin - ro) * invD;
    vec3 t1 = (bmax - ro) * invD;
    vec3 tmin = min(t0, t1);
    vec3 tmax = max(t0, t1);
    float tN = max(max(tmin.x, tmin.y), tmin.z);
    float tF = min(min(tmax.x, tmax.y), tmax.z);
    tHit = vec2(tN, tF);
    return (tF >= max(tN, 0.0));
}

void main() {
    vec2 ndc = vUV * 2.0 - 1.0;
    vec4 farP = uInvViewProj * vec4(ndc, 1.0, 1.0);
    vec3 pFar = farP.xyz / farP.w;

    vec3 ro = uCamPos;
    vec3 rd = normalize(pFar - ro);

    vec2 tHit;
    if (!intersectAABB(ro, rd, uBoxMin, uBoxMax, tHit)) {
        FragColor = vec4(0.0);
        return;
    }

    float t0 = max(tHit.x, 0.0);
    float t1 = tHit.y;
    float dt = (t1 - t0) / float(max(uSteps, 1));

    float alpha = 0.0;
    vec3 col = vec3(0.0);

    for (int i = 0; i < uSteps; ++i) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 p = ro + rd * t;

        vec3 uvw = (p - uBoxMin) / (uBoxMax - uBoxMin);
        float d = texture(uDensityTex, uvw).r;
        d = max(0.0, d * uDensityScale + uDensityBias);

        float a = 1.0 - exp(-uAbsorption * d * dt);
        vec3 sampleCol = vec3(1.0);

        float oneMinus = (1.0 - alpha);
        col += sampleCol * a * oneMinus;
        alpha += a * oneMinus;

        if (alpha > 0.995) break;
    }

    FragColor = vec4(col, alpha);
}
"""

POINTS_VERT = r"""
#version 330 core
layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;
uniform mat4 uMVP;
out vec3 vColor;
void main() {
    gl_Position = uMVP * vec4(inPos, 1.0);
    vColor = inColor;
    gl_PointSize = 4.0;
}
"""

POINTS_FRAG = r"""
#version 330 core
in vec3 vColor;
out vec4 FragColor;
void main() { FragColor = vec4(vColor, 1.0); }
"""

TOPDOWN_FRAG = r"""
#version 330 core
in vec2 vUV;
out vec4 FragColor;
uniform sampler2D uTex;
uniform float uGain;
void main() {
    float v = texture(uTex, vUV).r * uGain;
    v = clamp(v, 0.0, 1.0);
    FragColor = vec4(v, v, v, 1.0);
}
"""
