#include "Window.h"

#include <random>
#include <glm/glm.hpp>

#include <Iodine/Core/GraphicsAPI/Rand.h>



/// TODO:
/// - Abstract sphere to class which inherits from ObjectBase



Window::Window(const char* title, int width, int height)
{
	m_title = title;
	m_width = width;
	m_height = height;

	// Setup GLFW
	if (!glfwInit())
	{
		std::cout << "Failed to init GLFW3..." << std::endl;
		exit(1);
	}

	gl_window = glfwCreateWindow(width, height, title, NULL, NULL);
	
	if (!gl_window)
	{
		glfwTerminate();
		std::cout << "Failed to initialize window" << std::endl;
		exit(1);
	}
	
	glfwMakeContextCurrent(gl_window);
	gladLoadGL();

	// Set window color
	glClearColor(0.7, 0.8, 1, 1);
}

void Window::Begin()
{
	for (auto object : m_objects)
	{
		object->Begin();
	}
}

void Window::NewFrame()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

// Render ImGui onto the screen
void Window::Render()
{
	ImGui::SetNextWindowBgAlpha(0);
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	ImGui::Begin("Hello", (bool*)1, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDecoration);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("Model"))
		{
			if (ImGui::MenuItem("Render"))
			{
				m_shouldRender = true;
			}
			if (m_shouldRender)
			{
				if (ImGui::MenuItem("Stop"))
				{
					m_shouldRender = false;
				}
			}
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}

	if (m_shouldRender)
	{
		for (auto object : m_objects)
		{
			object->Render();
		}
	}

	char buffer[10];
	sprintf(buffer, "FPS: %i", m_fps);

	ImGui::Text(buffer);
	
	ImGui::End();

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}



void Window::End()
{
	for (auto object : m_objects)
	{
		object->End();
	}
}

void Window::SetFPS(int fps)
{
	this->m_fps = fps;
}


Window::~Window()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(gl_window);
	glfwTerminate();
}

