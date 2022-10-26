// Application.cc : Defines the entry point for the application.
//


#include "Application.h"
#include <Iodine/Core/Renderer/CubeObject.h>

#include <chrono>


Application::Application(const char* title, uint32_t width, uint32_t height)
{
	// Setup window
	m_window = new Window(title, width, height);

	for (int i = 0; i < 3; i++)
		m_window->AddObject(std::make_shared<Idn::CubeObject>());

	// Setup ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui_ImplGlfw_InitForOpenGL(m_window->gl_window, true);
	ImGui_ImplOpenGL3_Init();
}


void Application::Run()
{
	m_window->Begin();

	int fps = 0;
	int curFrames = 0;
	std::chrono::time_point timer = std::chrono::system_clock::now();

	while (!glfwWindowShouldClose(m_window->gl_window))
	{
		std::chrono::duration<double> elapsed = std::chrono::system_clock::now() - timer;

		if (elapsed.count() >= 1)
		{
			fps = curFrames;
			curFrames = 0;
			timer += std::chrono::milliseconds(1000);

			m_window->SetFPS(fps);
		}

		// Check if escape key was pressed
		if (glfwGetKey(m_window->gl_window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
			glfwSetWindowShouldClose(m_window->gl_window, true);
		
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		int width, height;
		glfwGetWindowSize(m_window->gl_window, &width, &height);

		m_window->SetWidth(width);
		m_window->SetHeight(height);
		
		m_window->NewFrame();
		m_window->Render();

		glfwSwapBuffers(m_window->gl_window);
		glfwPollEvents();
		
		curFrames++;
	}

	m_window->End();
}


Application::~Application()
{
	delete m_window;
}

