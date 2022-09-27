#pragma once

#include <vector>
#include <memory>
#include <imgui.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <Core/Renderer/Object.h>


class Window
{
public:
	GLFWwindow* gl_window;
	std::vector<std::shared_ptr<Idn::ObjectBase>> m_objects;
	
	Window(const char* title, int width=500, int height=400);

	// Create a new ImGui frame (sep)
	void Begin();
	void NewFrame();
	void Render();
	void End();

	int GetWidth() { return m_width; }
	void SetWidth(int width) { this->m_width = width; }

	void SetFPS(int fps);

	int GetHeight() { return m_height; }
	void SetHeight(int height) { this->m_height = height; }
	void AddObject(std::shared_ptr<Idn::ObjectBase> object) { m_objects.push_back(object); }

	~Window();
private:
	// Pointer to the main opaque window object
	const char* m_title;
	int m_width, m_height, m_fps = 0;

	bool m_shouldRender = false;
};

