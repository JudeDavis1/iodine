
#include <imgui.h>


// local
#include "SphereObject.h"
#include "../Utils/Image.h"



void SphereObject::Begin() {}
void SphereObject::End() {}


void SphereObject::Render()
{
	GLuint txtr;
	float d = m_radius * 2;
	uint32_t* img_data = new uint32_t[d * d];

	for (uint32_t y = 0; y < d; y++)
	{
		for (uint32_t x = 0; x < d; x++)
		{
			glm::vec2 coords = { (float)x / d, (float)y / d };
			coords = coords * 2.0f - 1.0f;

			glm::vec4 color = glm::clamp(m_PerPixel(coords), glm::vec4(0.0f), glm::vec4(1.0f));
			img_data[x + y * (int)d] = Idn::ToRGBA(color);
		}
	}

	Idn::CreateTexture((unsigned char*)img_data, d, d, &txtr);
	ImGui::Image((void*)(intptr_t)txtr, ImVec2(d, d));
}

glm::vec4 SphereObject::m_PerPixel(glm::vec2 coords)
{
	uint8_t r = (uint8_t)(coords.x * 255.0f);
	uint8_t g = (uint8_t)(coords.y * 255.0f);

	float radius = 0.7;
	glm::vec3 rayOrigin(0.0f, 0.0f, 1.0f);
	glm::vec3 rayDirection(coords.x, coords.y, -1.0f);

	float a = glm::dot(rayDirection, rayDirection);
	float b = 2.0f * glm::dot(rayOrigin, rayDirection);
	float c = glm::dot(rayOrigin, rayOrigin) - radius * radius;
	float discriminant = b * b - 4.0f * a * c;

	if (discriminant < 0.0f)
		return glm::vec4(0, 0, 0, 0);

	float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a);
	float t1 = (-b - glm::sqrt(discriminant)) / (2.0f * a);  // Closest hit

	glm::vec3 hitPoint = rayOrigin + rayDirection * t1;
	glm::vec3 normal = glm::normalize(hitPoint);

	glm::vec3 lightingDirection = glm::normalize(glm::vec3(-1, 1, -1));
	float d = glm::max(glm::dot(normal, -lightingDirection), 0.2f);

	glm::vec3 sphereColor(0.2, 0, 1);
	sphereColor *= d;

	return glm::vec4(sphereColor, 1.0f);
}

