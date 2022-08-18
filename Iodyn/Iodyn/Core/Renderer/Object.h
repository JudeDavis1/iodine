#include <iostream>


// Abstract class for all objects in a rendering window e.g: A sphere
class ObjectBase
{
public:
	virtual ObjectBase() = default;
	
	virtual void Begin() {}
	virtual void NewFrame()
	{
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	virtual void Render()
	{
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
	}

	virtual void Update() = default;
	virtual void End() {}
};
