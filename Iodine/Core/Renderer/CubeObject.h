#pragma once

#include "Object.h"
#include "../Shaders/Shader.h"
#include "Camera.h"

// OpenGL mathematics and vector/matrix transforms
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>


namespace Idn
{
	class CubeObject : public ObjectBase
	{
	public:
		glm::mat4 projection = glm::mat4(1);

		CubeObject();
		~CubeObject();

		void Begin() override;
		void Render() override;
		void End() override;
		void SetTexture() override;

		inline void SetPosition(const glm::vec3& new_pos)
		{
			m_model = glm::translate(m_model, new_pos);
		}

		float GetSize() { return m_size; }
		inline void SetSize(int new_size) { m_size = new_size; }
	private:
		float m_size = 0.5;
		GLuint m_VAO, m_VBO, m_tex;
		std::vector<GLfloat> m_verticies;
		std::shared_ptr<Idn::Shader> m_shader;
	};

}