#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 tex;

out vec2 outTex;

uniform mat4 view;
uniform mat4 model;
uniform mat4 transform;

void main()
{
	gl_Position = transform * vec4(position + 0.1, 1.0f);
	outTex = vec2(tex.x, 1.0 - tex.y);
}
