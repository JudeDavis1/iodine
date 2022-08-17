#include "Window.h"
#include <random>
#include <Iodyn/Core/Utils/Rand.h>
#include <Iodyn/Core/Utils/Image.h>


/// TODO:
/// - Add image creation and convert to ImGui texture




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

uint32_t PerPixel(ImVec2 coords)
{
	uint8_t r = (uint8_t)(coords.x * 255.0f);
	uint8_t g = (uint8_t)(coords.y * 255.0f);


	return 0xff000000 | (g << 8) | r;
}

void RenderSomething()
{
	int width = 400;
	int height = 400;
	GLuint txtr;
	uint32_t* img_data = new uint32_t[width * height];


	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			ImVec2 coords = ImVec2((float)x / (float)width, (float)y / (float)height);
			img_data[x + y * width] = PerPixel(coords);
		}
	}

	Idn::CreateTexture((unsigned char*)img_data, width, height, &txtr);
	ImGui::Image((void*)(intptr_t)txtr, ImVec2(width, height));
}

void Window::Update()
{
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
	ImGui::Begin("Hello", (bool*)1, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDecoration);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::BeginMenu("Scan"))
		{
			if (ImGui::MenuItem("Begin"))
			{
				std::cout << "ok" << std::endl;
			}
			if (ImGui::MenuItem("Stop")) {}
			ImGui::EndMenu();
		}

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
		RenderSomething();

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

