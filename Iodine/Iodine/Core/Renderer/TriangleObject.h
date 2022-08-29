#pragma once

#include "Object.h"
#include "../Shaders/Shader.h"


namespace Idn
{
	class TriangleObject : public ObjectBase
	{
	public:
		TriangleObject();
		~TriangleObject();

		void Begin() override;
		void Render() override;
		void End() override;
	private:
		std::shared_ptr<Idn::Shader> m_shader;
	};

}