#pragma once

#include <vector>
#include <iostream>

#include "Object.h"
#include "Camera.h"


class Renderer
{
public:
	Idn::Camera camera = Idn::Camera(glm::vec3(0));

	Renderer(GLFWwindow* window);

	void Begin();
	void NewFrame();
	void Render();
	void AddObject(std::shared_ptr<Idn::ObjectBase> object) { m_objects.push_back(object); }
	std::vector<std::shared_ptr<Idn::ObjectBase>>& GetObjects() { return m_objects; }
	void End();

	~Renderer();
private:
	std::vector<std::shared_ptr<Idn::ObjectBase>> m_objects;
};

