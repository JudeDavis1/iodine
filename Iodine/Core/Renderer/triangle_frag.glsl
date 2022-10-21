#version 330 core

in vec3 col;
in vec2 outTex;

out vec4 new_col;

// Texture sampler
uniform sampler2D txtr;

void main()
{
	new_col = texture(txtr, outTex);
}
