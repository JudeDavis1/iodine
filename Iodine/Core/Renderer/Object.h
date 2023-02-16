#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <glm/glm.hpp>

#include "../Shaders/Shader.h"


namespace Idn
{
	// Abstract class for all objects in a rendering window e.g: A sphere
	class ObjectBase
	{
	public:
		uint32_t* winWIDTH = nullptr;
		uint32_t* winHEIGHT = nullptr;

		ObjectBase() = default;

		virtual void Begin() = 0;
		virtual void Render() = 0;
		virtual void End() = 0;
		virtual void SetTexture() = 0;
		void SetCameraPos(const glm::mat4 pos) { m_view = pos; }

		virtual ~ObjectBase() = default;
	protected:
		std::vector<GLfloat> m_verticies;
		std::shared_ptr<Shader> m_shader = nullptr;
		glm::mat4 m_view = glm::mat4(1);
		glm::mat4 m_model = glm::mat4(1);
	};
}
