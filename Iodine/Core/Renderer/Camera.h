#pragma once

#include "Object.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace Idn
{
	enum MovementDirection
	{
		LEFT, RIGHT, FD, BD
	};

	class Camera
	{
	public:
		Camera(const glm::vec3& pos)
		{
			this->m_camera_pos = pos;
		}

		void Begin() {}
		void End() {}
		void Render() {}
		void SetTexture() {}

		void SetPosition(const glm::vec3& pos)
		{
			this->m_view_mat = glm::translate(glm::mat4(1), pos);
			this->m_camera_pos = pos;
		}
		
		const glm::mat4& GetPosition()
		{
			return this->m_view_mat;
		}

	private:
		glm::mat4 m_view_mat;
		glm::vec3 m_camera_pos;
	};

};