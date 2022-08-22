#pragma once

#include "Object.h"

#include <iostream>
#include <glm/glm.hpp>
#include <glad/glad.h>
#include <GLFW/glfw3.h>



class SphereObject: public ObjectBase
{
public:
	SphereObject() {}
	
	void Begin() override;
	void Render() override;
	void End() override;

	float GetRadius() { return m_radius; }
	void SetRadius(float r) { m_radius = r; }
	
	~SphereObject() {}

private:
	float m_radius = 200;

	glm::vec4 m_PerPixel(glm::vec2 coords);
};

	