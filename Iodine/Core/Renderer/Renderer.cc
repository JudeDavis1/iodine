#include "Renderer.h"
#include "CubeObject.h"

#include <assert.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>



Renderer::Renderer(GLFWwindow* window) {
	assert(window != nullptr);
}

void Renderer::Begin() {
	// Begin other objects
	for (auto object : m_objects) object->Begin();

	// Set default camera position
	glm::vec3 default_camera_pos = glm::vec3(0, 0, -2);
	camera.SetPosition(default_camera_pos);
}

void Renderer::NewFrame() {
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
}

void Renderer::Render() {
	for (auto object : m_objects) {
		object->SetCameraPos(this->camera.GetPosition());
		object->Render();
	}
}

void Renderer::End() {
	for (auto object : m_objects) object->End();
}

Renderer::~Renderer() {
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
}

