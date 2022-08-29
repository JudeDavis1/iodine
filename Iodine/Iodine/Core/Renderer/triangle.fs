#version 330 core

in vec3 col;
out vec4 new_col;

void main()
{
	new_col = vec4(col, 1.0f);
}