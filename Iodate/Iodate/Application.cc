// Application.cc : Defines the entry point for the application.
//


#include "Application.h"



Application::Application(const char* title, int width, int height)
{
	m_window = new Window(title, width, height);
	
	// Setup ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(0, 0));

	ImGui_ImplGlfw_InitForOpenGL(m_window->gl_window, true);
	ImGui_ImplOpenGL3_Init();
}


void Application::Run()
{
	m_window->Begin();

	while (!glfwWindowShouldClose(m_window->gl_window))
	{
		// Check if escape key was pressed
		if (glfwGetKey(m_window->gl_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(m_window->gl_window, true);
		
		glClear(GL_COLOR_BUFFER_BIT);

		int width, height;
		glfwGetWindowSize(m_window->gl_window, &width, &height);

		m_window->SetWidth(width);
		m_window->SetHeight(height);
		
		m_window->NewFrame();
		m_window->Update();
		m_window->Render();

		glfwSwapBuffers(m_window->gl_window);
		glfwPollEvents();
	}

	m_window->End();
}


Application::~Application()
{
	delete m_window;
}

