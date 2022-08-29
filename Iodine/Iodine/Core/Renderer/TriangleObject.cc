#include "TriangleObject.h"


namespace Idn
{
	TriangleObject::TriangleObject()
	{
		m_shader = std::make_shared<Idn::Shader>("C:\\Users\\juded\\OneDrive\\Documents\\Projects\\iodine\\Iodine\\Iodine\\Core\\Renderer\\triangle.vs", "C:\\Users\\juded\\OneDrive\\Documents\\Projects\\iodine\\Iodine\\Iodine\\Core\\Renderer\\triangle.fs");
	}

	void TriangleObject::Begin()
	{
		GLfloat vertices[] = {
			 // Positions            // Colors
			 0.5f, -0.5f, 0.0f,      1.0f, 0.0f, 0.0f,
			-0.5f, -0.5f, 0.0f,      0.0f, 1.0f, 0.0f,
			 0.0f,  0.5f, 0.0f,      0.0f, 0.0f, 1.0f,
		};

		GLuint vbo, vao;
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);

		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		// Position attributes
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GL_FLOAT), (void*) 0);
		glEnableVertexAttribArray(0);

		// Color attributes
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(GL_FLOAT), (void*) 0);
		glEnableVertexAttribArray(1);


		m_shader->Compile();
		m_shader->Use();
	}

	void TriangleObject::End() {}

	void TriangleObject::Render()
	{
		glDrawArrays(GL_TRIANGLES, 0, 3);
	}

	TriangleObject::~TriangleObject()
	{

	}
}
