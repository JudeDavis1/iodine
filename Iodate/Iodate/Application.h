// Iodate.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <imgui.h>
#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "Window.h"


class Application
{
public:
	// Run the application loop
	void Run();

	// Init GLFW and window
	Application(const char* title, int width = 500, int height = 400);
	~Application();
private:
	// Pointer to the main opaque window object
	Window* m_window;
	std::string m_title;
};


