#include "TriangleObject.h"
#include "Core/GraphicsAPI/Image.h"

#include <stb/stb_image.h>


namespace Idn
{
	TriangleObject::TriangleObject()
	{
		m_shader = std::make_shared<Idn::Shader>("C:\\Users\\juded\\OneDrive\\Documents\\Projects\\iodine\\Iodine\\Core\\Renderer\\triangle_vtx.glsl", "C:\\Users\\juded\\OneDrive\\Documents\\Projects\\iodine\\Iodine\\Core\\Renderer\\triangle_frag.glsl");
	}

	void TriangleObject::Begin()
	{
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		GLfloat vertices[] = {
			 // Positions            // Colors               // Texture coords
			 0.5f,   0.5f, 0.0f,     1.0f, 0.0f, 0.0f,       1.0f, 1.0f,
			 0.5f,  -0.5f, 0.0f,     0.0f, 1.0f, 0.0f,       1.0f, 0.0f,
			-0.5f,  -0.5f, 0.0f,     0.0f, 0.0f, 1.0f,       0.0f, 0.0f,
			-0.5f,   0.5f, 0.0f,     1.0f, 1.0f, 0.0f,       0.0f, 1.0f,
		};
		GLuint indicies[] = {
			0, 1, 3,
		};
		
		glGenVertexArrays(1, &m_VAO);
		glGenBuffers(1, &m_VBO);
		glGenBuffers(1, &m_EBO);

		glBindVertexArray(m_VAO);
		glBindBuffer(GL_ARRAY_BUFFER, m_VBO);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_EBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indicies), indicies, GL_STATIC_DRAW);

		int attribIdx = 0;

		// Position attributes
		glVertexAttribPointer(attribIdx, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*) 0);
		glEnableVertexAttribArray(attribIdx);

		attribIdx++;

		// Color attributes
		glVertexAttribPointer(attribIdx, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GL_FLOAT), (void*) (3 * sizeof(GLfloat)));
		glEnableVertexAttribArray(attribIdx);

		attribIdx++;

		// Texture attributes
		glVertexAttribPointer(attribIdx, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GL_FLOAT), (void*)(6 * sizeof(GLfloat)));
		glEnableVertexAttribArray(attribIdx);

		m_shader->Compile();

		SetTexture();
	}


	void TriangleObject::Render()
	{
		m_shader->Use();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_tex);
		glUniform1i(glGetUniformLocation(m_shader->GetProgram(), "t"), 0);

		glBindVertexArray(m_VAO);
		
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		
		glBindVertexArray(0);
	}

	void TriangleObject::SetTexture()
	{
		ImageFmt fmt = Idn::Load("C:\\Users\\juded\\Downloads\\tiger.jpg", 4);

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

	void TriangleObject::End()
	{
		glDeleteVertexArrays(1, &m_VAO);
		glDeleteBuffers(1, &m_VBO);
	}

	TriangleObject::~TriangleObject()
	{
	}
}
