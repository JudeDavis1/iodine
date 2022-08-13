#include "Window.h"


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
	glClearColor(0, 0.5, 0.7, 1);
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
	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Window::Update()
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImVec2(m_width, m_height));
	ImGui::Begin("Hello", (bool*)1, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);

	ImGui::Text("Ok");

	ImGui::End();
}


Window::~Window()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(gl_window);
	glfwTerminate();
}