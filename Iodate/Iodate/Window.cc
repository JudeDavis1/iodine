#include "Window.h"

#include <random>
#include <glm/glm.hpp>

#include <Iodine/Core/GraphicsAPI/Rand.h>



/// TODO:
/// - Abstract sphere to class which inherits from ObjectBase


Window::Window(const char* title, uint32_t width, uint32_t height) {
	m_title = title;
	m_width = width;
	m_height = height;


	// Setup GLFW
	if (!glfwInit())
	{
		std::cout << "Failed to init GLFW3..." << std::endl;
		exit(1);
	}

#ifdef __APPLE__
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	gl_window = glfwCreateWindow(width, height, title, NULL, NULL);
	m_renderer = std::make_shared<Renderer>(gl_window);
	
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

void Window::Begin() {
	// Update width, height pointers in each object
	for (auto object : m_renderer->GetObjects())
	{
		object->winWIDTH = &m_width;
		object->winHEIGHT = &m_height;
	}
	m_renderer->Begin();
}

void Window::NewFrame() {
	m_renderer->NewFrame();
}

// Render ImGui onto the screen
void Window::Render() {
	ImGui::SetNextWindowBgAlpha(0);
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	ImGui::Begin("Hello", (bool*)true, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDecoration);

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

	if (m_shouldRender) m_renderer->Render();

	char buffer[10];
	snprintf(buffer, 10, "FPS: %i", m_fps);

	ImGui::Text(buffer);
	ImGui::End();
	ImGui::Render();

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::End() {
	m_renderer->End();
}

void Window::SetFPS(int fps) {
	this->m_fps = fps;
}


Window::~Window() {
	glfwDestroyWindow(gl_window);
	glfwTerminate();
}

