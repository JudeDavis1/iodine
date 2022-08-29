#pragma once

#include <iostream>

#include "../Shaders/Shader.h"


namespace Idn
{
	// Abstract class for all objects in a rendering window e.g: A sphere
	class ObjectBase
	{
	public:
		ObjectBase() = default;

		virtual void Begin() = 0;
		virtual void Render() = 0;
		virtual void End() = 0;

		virtual ~ObjectBase() = default;
	protected:
		std::shared_ptr<Shader> m_shader = nullptr;
	};
}
