#include "CubeObject.h"
#include "../GraphicsAPI/Rand.h"
#include "../GraphicsAPI/Image.h"


// To load and free images
#include <stb/stb_image.h>
#include <glm/gtc/matrix_transform.hpp>


namespace Idn
{
	CubeObject::CubeObject()
	{
		m_shader = std::make_shared<Idn::Shader>(
			"/Users/judedavis/Library/CloudStorage/OneDrive-Personal/Documents/Projects/iodine/Iodine/Core/Renderer/triangle_vtx.glsl",
			"/Users/judedavis/Library/CloudStorage/OneDrive-Personal/Documents/Projects/iodine/Iodine/Core/Renderer/triangle_frag.glsl"
		);
		m_verticies = std::vector<GLfloat>{
			// Positions			// Texture coords
		   -0.5f, -0.5f, -0.5f,		0.0f, 0.0f,
			0.5f, -0.5f, -0.5f,		1.0f, 0.0f,
			0.5f,  0.5f, -0.5f,		1.0f, 1.0f,
			0.5f,  0.5f, -0.5f,		1.0f, 1.0f,
		   -0.5f,  0.5f, -0.5f,		0.0f, 1.0f,
		   -0.5f, -0.5f, -0.5f,		0.0f, 0.0f,

		   -0.5f, -0.5f,  0.5f,		0.0f, 0.0f,
			0.5f, -0.5f,  0.5f,		1.0f, 0.0f,
			0.5f,  0.5f,  0.5f,		1.0f, 1.0f,
			0.5f,  0.5f,  0.5f,		1.0f, 1.0f,
		   -0.5f,  0.5f,  0.5f,		0.0f, 1.0f,
		   -0.5f, -0.5f,  0.5f,		0.0f, 0.0f,

		   -0.5f,  0.5f,  0.5f,		1.0f, 0.0f,
		   -0.5f,  0.5f, -0.5f,		1.0f, 1.0f,
		   -0.5f, -0.5f, -0.5f,		0.0f, 1.0f,
		   -0.5f, -0.5f, -0.5f,		0.0f, 1.0f,
		   -0.5f, -0.5f,  0.5f,		0.0f, 0.0f,
		   -0.5f,  0.5f,  0.5f,		1.0f, 0.0f,

			0.5f,  0.5f,  0.5f,		1.0f, 0.0f,
			0.5f,  0.5f, -0.5f,		1.0f, 1.0f,
			0.5f, -0.5f, -0.5f,		0.0f, 1.0f,
			0.5f, -0.5f, -0.5f,		0.0f, 1.0f,
			0.5f, -0.5f,  0.5f,		0.0f, 0.0f,
			0.5f,  0.5f,  0.5f,		1.0f, 0.0f,

		   -0.5f, -0.5f, -0.5f,		0.0f, 1.0f,
			0.5f, -0.5f, -0.5f,		1.0f, 1.0f,
			0.5f, -0.5f,  0.5f,		1.0f, 0.0f,
			0.5f, -0.5f,  0.5f,		1.0f, 0.0f,
		   -0.5f, -0.5f,  0.5f,		0.0f, 0.0f,
		   -0.5f, -0.5f, -0.5f,		0.0f, 1.0f,

		   -0.5f,  0.5f, -0.5f,		0.0f, 1.0f,
			0.5f,  0.5f, -0.5f,		1.0f, 1.0f,
			0.5f,  0.5f,  0.5f,		1.0f, 0.0f,
			0.5f,  0.5f,  0.5f,		1.0f, 0.0f,
		   -0.5f,  0.5f,  0.5f,		0.0f, 0.0f,
		   -0.5f,  0.5f, -0.5f,		0.0f, 1.0f
		};
	}

	void CubeObject::Begin()
	{
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		GLuint indicies[] = {
			0, 1, 2,
		};

		// Copy the data from m_verticies to a GLfloat pointer
		GLfloat* vert_ptr = new GLfloat[m_verticies.size() * sizeof(GLfloat)];
		std::copy(m_verticies.begin(), m_verticies.end(), vert_ptr);

		glGenVertexArrays(1, &m_VAO);
		glGenBuffers(1, &m_VBO);

		glBindVertexArray(m_VAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * m_verticies.size(), vert_ptr, GL_STATIC_DRAW);
		delete[] vert_ptr;

		int attribIdx = 0;

		// Position attributes
		glVertexAttribPointer(attribIdx, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*) 0);
		glEnableVertexAttribArray(attribIdx);

		attribIdx++;

		// Texture attributes
		glVertexAttribPointer(attribIdx, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GL_FLOAT), (void*)(3 * sizeof(GLfloat)));
		glEnableVertexAttribArray(attribIdx);

		m_shader->Compile();

		SetTexture();

		float z_near = 0.1f;
		float z_far = 100.0f;
		projection = glm::perspective(45.0f, (GLfloat) *winWIDTH / (GLfloat)*winHEIGHT, z_near, z_far);
		this->SetPosition(glm::vec3(0, 0, -5));
	}


	void CubeObject::Render()
	{
		m_shader->Use();

		// Camera matrix
		m_model = glm::mat4(1);
		m_model = glm::rotate(m_model, (float)glfwGetTime() * 1, glm::vec3(1, 1, 1));

		// Upload projection to opengl
		GLint viewLoc = glGetUniformLocation(m_shader->GetProgram(), "view");
		GLint modelLoc = glGetUniformLocation(m_shader->GetProgram(), "model");
		GLint projLoc = glGetUniformLocation(m_shader->GetProgram(), "projection");

		glUniformMatrix4fv(viewLoc,  1, GL_FALSE, glm::value_ptr(m_view));
		glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(m_model));
		glUniformMatrix4fv(projLoc,  1, GL_FALSE, glm::value_ptr(projection));
		
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glUniform1i(glGetUniformLocation(m_shader->GetProgram(), "txtr"), 0);

		// Draw
		glBindVertexArray(m_VAO);
		glDrawArrays(GL_TRIANGLES, 0, 6 * 6);
		glBindVertexArray(0);
	}

	void CubeObject::SetTexture()
	{
		ImageFmt fmt = Idn::Load("/Users/judedavis/Downloads/texture.jpg", 4);

		GLuint tmp_tx;
		Idn::CreateTexture(fmt.data, fmt.width, fmt.height, &tmp_tx, []() {
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		});

		glGenerateMipmap(GL_TEXTURE_2D);
		stbi_image_free((void*) fmt.data);
		glBindTexture(GL_TEXTURE_2D, 0);

		this->m_tex = tmp_tx;
	}

	void CubeObject::End()
	{
		glDeleteVertexArrays(1, &m_VAO);
		glDeleteBuffers(1, &m_VBO);
	}

	CubeObject::~CubeObject() {}
}
