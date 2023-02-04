#include "Image.h"
#include "Rand.h"

#define STB_IMAGE_IMPLEMENTATION

#include <stb/stb_image.h>


namespace Idn
{

	bool CreateTexture(const uint8_t* image_data,
		int width,
		int height,
		GLuint* out_texture,
		void (*texParams)(void),  // void Function which sets up texture parameters
		int target_dim)
	{
		if (image_data == nullptr || image_data == NULL)
		{
			std::cout << "ERROR: The image provided is NULL!" << std::endl;
			return false;
		}

		// Generate a texture
		GLuint image_texture;
		glGenTextures(1, &image_texture);
		glBindTexture(target_dim, image_texture);

		if (image_texture == NULL)
			return false;

		// Setup filtering params for display
		texParams();

		// Upload pixels to texture
		glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data);
		*out_texture = image_texture;

		return true;
	}


	uint32_t ToRGBA(glm::vec4& color)
	{
		uint8_t r = (uint8_t)(color.r * 255.0f);
		uint8_t g = (uint8_t)(color.g * 255.0f);
		uint8_t b = (uint8_t)(color.b * 255.0f);
		uint8_t a = (uint8_t)(color.a * 255.0f);

		uint32_t result = (a << 24) | (b << 16) | (g << 8) | r;

		return result;
	}
	
	// Default texture parameters
	void _TexParamsDef()
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	int rndNi(int low, int high)
	{
		std::default_random_engine _generator;
		std::uniform_int_distribution<int> dist(low, high);

		return dist(_generator);
	}

	ImageFmt Load(const char* filename, int preferred_channels)
	{
		int width, height, channels;
		const uint8_t* image_data = stbi_load(filename, &width, &height, &channels, preferred_channels);

		ImageFmt img;
		img.width = width;
		img.height = height;
		img.data = image_data;
		img.channels = channels;

		return img;
	}

}
