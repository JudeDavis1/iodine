#include "Renderer.h"
#include "CubeObject.h"

#include <assert.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>



Renderer::Renderer(GLFWwindow* window)
{
	assert(window != nullptr);
}

void Renderer::Begin()
{
	for (auto object : m_objects)
		object->Begin();
}

void Renderer::NewFrame()
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void Renderer::Render()
{
	for (auto object : m_objects)
		object->Render();
}

void Renderer::End()
{
	for (auto object : m_objects)
		object->End();
}

Renderer::~Renderer()
{
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

