#pragma once

#include <imgui.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>


class Window
{
public:
	GLFWwindow* gl_window;
	
	Window(const char* title, int width=500, int height=400);

	// Create a new ImGui frame (sep)
	void Begin();
	void NewFrame();
	void Render();
	void Update();
	void End();

	int GetWidth() { return m_width; }
	void SetWidth(int width) { this->m_width = width; }

	int GetHeight() { return m_height; }
	void SetHeight(int height) { this->m_height = height; }

	~Window();
private:
	// Pointer to the main opaque window object
	const char* m_title;
	int m_width, m_height;

	bool m_shouldRender = false;
};

