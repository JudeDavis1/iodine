#pragma once

#include <vector>
#include <iostream>
#include <GLFW/glfw3.h>

#include "Object.h"

class Renderer
{
public:

	Renderer(GLFWwindow* window);

	void Begin();
	void NewFrame();
	void Render();
	void AddObject(std::shared_ptr<Idn::ObjectBase> object) { m_objects.push_back(object); }
	void End();

	~Renderer();
private:
	std::vector<std::shared_ptr<Idn::ObjectBase>> m_objects;
};

