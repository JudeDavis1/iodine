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
#include <Core/Renderer/Camera.h>
#include <Core/Renderer/Renderer.h>


class Window
{
public:
	GLFWwindow* gl_window;
	
	Window(const char* title, uint32_t width=600, uint32_t height=400);

	// Create a new ImGui frame (sep)
	void Begin();
	void NewFrame();
	void Render();
	void End();

	int GetWidth() { return m_width; }
	void SetWidth(uint32_t width) { this->m_width = width; }

	void SetFPS(int fps);

	int GetHeight() { return m_height; }
	void SetHeight(uint32_t height) { this->m_height = height; }
	void AddObject(std::shared_ptr<Idn::ObjectBase> object) { m_renderer->AddObject(object); }
	std::shared_ptr<Renderer> GetRenderer() { return m_renderer; }

	~Window();
private:
	// Pointer to the main opaque window object
	const char* m_title;
	bool m_shouldRender = false;
	uint32_t m_width, m_height, m_fps = 0;
	std::shared_ptr<Renderer> m_renderer = nullptr;
};

