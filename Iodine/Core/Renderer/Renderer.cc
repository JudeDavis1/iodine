#include <assert.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include "Renderer.h"
#include "TriangleObject.h"


Renderer::Renderer(GLFWwindow* window)
{
	assert(window != nullptr);

	// Setup ImGui
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();
}


void Renderer::Render()
{

}


Renderer::~Renderer()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

