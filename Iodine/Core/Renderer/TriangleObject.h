#pragma once

#include "Object.h"
#include "../Shaders/Shader.h"


namespace Idn
{
	class TriangleObject : public ObjectBase
	{
	public:
		TriangleObject();
		~TriangleObject();

		void Begin() override;
		void Render() override;
		void End() override;
		void SetTexture() override;
	private:
		GLuint m_VAO, m_VBO, m_EBO, m_tex;
		std::shared_ptr<Idn::Shader> m_shader;
	};

}