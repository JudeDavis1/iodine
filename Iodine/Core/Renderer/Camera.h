#pragma once

#include "Object.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>


namespace Idn
{
	enum MovementDirection
	{
		LEFT=GLFW_KEY_A, RIGHT=GLFW_KEY_D, FD=GLFW_KEY_W, BD=GLFW_KEY_S
	};

	class Camera
	{
	public:
		Camera(const glm::vec3& pos)
		{
			this->m_camera_pos = pos;
		}

		void SetPosition(const glm::vec3& pos)
		{
			this->m_view_mat = glm::translate(glm::mat4(1), pos);
			this->m_camera_pos = pos;
		}
		
		const glm::mat4& GetPosition()
		{
			return this->m_view_mat;
		}

		const void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
		{
			// The amount to move
			float delta = 0.03;
			switch (key)
			{
			case MovementDirection::FD:
				this->SetPosition(glm::vec3(m_camera_pos.x, m_camera_pos.y, m_camera_pos.z + delta));
				break;
			case MovementDirection::LEFT:
				this->SetPosition(glm::vec3(m_camera_pos.x + delta, m_camera_pos.y, m_camera_pos.z));
				break;
			case MovementDirection::BD:
				this->SetPosition(glm::vec3(m_camera_pos.x, m_camera_pos.y, m_camera_pos.z - delta));
				break;
			case MovementDirection::RIGHT:
				this->SetPosition(glm::vec3(m_camera_pos.x - delta, m_camera_pos.y, m_camera_pos.z));
				break;
			}
		}

	private:
		glm::mat4 m_view_mat;
		glm::vec3 m_camera_pos;
	};

};