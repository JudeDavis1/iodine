#version 440 core

in vec3 col;
in vec2 tex;

out vec4 new_col;

// Texture sampler
uniform sampler2D t;

void main()
{
	new_col = texture(t, tex);
}