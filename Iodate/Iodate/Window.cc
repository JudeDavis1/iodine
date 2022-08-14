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

void Window::Begin()
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
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
	ImGui::SetNextWindowSize(ImVec2(m_width / 2, m_height / 2));
	ImGui::Begin("Hello", (bool*)1, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_MenuBar);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("Scan"))
		{
			if (ImGui::MenuItem("Begin"))
			{
				
			}
			if (ImGui::MenuItem("Stop")) {}
			ImGui::EndMenu();
		}

		ImGui::EndMenuBar();
	}

	ImGui::End();
}

void Window::End() {}


Window::~Window()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	glfwDestroyWindow(gl_window);
	glfwTerminate();
}

