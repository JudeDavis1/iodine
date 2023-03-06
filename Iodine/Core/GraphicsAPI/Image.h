#pragma once

#include <iostream>
#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>


namespace Idn {

	struct ImageFmt {
		const uint8_t* data;

		int channels;
		int width, height;
	};
	
	void _TexParamsDef();
	bool CreateTexture(const uint8_t* image_data,
		int width,
		int height,
		GLuint* out_texture,
		void (*texParams)(void) = _TexParamsDef,  // void Function which sets up texture parameters
		int target_dim = GL_TEXTURE_2D);

	uint32_t ToRGBA(glm::vec4& color);
	ImageFmt Load(const char* filename, int preferred_channels=3);
}

