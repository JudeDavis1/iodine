#include "Window.h"
#include <random>
#include <glm/glm.hpp>
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

glm::vec4 PerPixel(glm::vec2 coords, int width=500, int height=500)
{
	uint8_t r = (uint8_t)(coords.x * 255.0f);
	uint8_t g = (uint8_t)(coords.y * 255.0f);

	float radius = 0.5f;
	glm::vec3 rayOrigin(0.0f, 0.0f, 1.0f);
	glm::vec3 rayDirection(coords.x, coords.y, -1.0f);

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;

	if (discriminant < 0.0f)
		return glm::vec4(0, 0, 0, 1);

	float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a);
	float t1 = (-b - glm::sqrt(discriminant)) / (2.0f * a);  // Closest hit

	glm::vec3 hitPoint = rayOrigin + rayDirection * t1;
	glm::vec3 normal = glm::normalize(hitPoint);

	glm::vec3 lightingDirection = glm::normalize(glm::vec3(-1, -1, -1));
	float d = glm::max(glm::dot(normal, -lightingDirection), 0.2f);

	glm::vec3 sphereColor(0.5, 0, 1);
	sphereColor *= d;

	return glm::vec4(sphereColor, 1.0f);
}

void RenderSomething()
{
	GLuint txtr;
	int width = 500;
	int height = 500;
	uint32_t* img_data = new uint32_t[width * height];

	for (uint32_t y = 0; y < height; y++)
	{
		for (uint32_t x = 0; x < width; x++)
		{
			glm::vec2 coords = { (float)x / (float)width, (float)y / (float)height };
			coords = coords * 2.0f - 1.0f;

			glm::vec4 color = glm::clamp(PerPixel(coords), glm::vec4(0.0f), glm::vec4(1.0f));
			img_data[x + y * width] = Idn::ToRGBA(color);
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

