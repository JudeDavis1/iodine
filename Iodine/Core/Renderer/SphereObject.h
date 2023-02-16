#pragma once

#include "Object.h"

#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/string_cast.hpp>


namespace Idn
{
	class SphereObject : public ObjectBase
	{
	public:
		SphereObject();
		~SphereObject();

		void Begin() override;
		void Render() override;
		void End() override;
		void SetTexture() override;

		float GetRadius() { return m_radius; }
		void SetRadius(float r) { m_radius = r; }

		inline void SetPosition(const glm::vec3& new_pos)
		{
			m_model = glm::translate(glm::mat4(0), new_pos);
		}

	private:
		float m_radius = 200;
		std::shared_ptr<Idn::Shader> m_shader = nullptr;

		glm::vec4 m_PerPixel(glm::vec2 coords);
	};

}