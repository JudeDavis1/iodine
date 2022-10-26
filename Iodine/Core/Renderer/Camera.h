#pragma once

#include <Object.h>
#include <glm/glm.hpp>


namespace Idn
{
	enum MovementDirection
	{
		LEFT, RIGHT, FD, BD
	};

	class Camera
	{
	public:
		Camera(glm::vec3& pos)
		{
			this->camera_pos = pos;
		}

	private:
		glm::vec3 camera_pos;
	};

};