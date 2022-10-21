#pragma once

#include "Object.h"

#include <iostream>
#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>


namespace Idn
{
	class SphereObject : public ObjectBase
	{
	public:
		SphereObject();
		~SphereObject() {}

		void Begin() override;
		void Render() override;
		void End() override;

		float GetRadius() { return m_radius; }
		void SetRadius(float r) { m_radius = r; }

	private:
		float m_radius = 200;
		std::shared_ptr<Idn::Shader> m_shader = nullptr;

		glm::vec4 m_PerPixel(glm::vec2 coords);
	};

}