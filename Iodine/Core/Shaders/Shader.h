#pragma once


#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

namespace Idn
{
	class Shader
	{
	public:
		Shader(std::string vtx_path, std::string frag_path);
		~Shader();

		bool Compile();
		GLuint GetProgram() { return m_program; }

		void Use();
	private:
		GLuint m_program;
		std::string m_vtxSrc, m_fragSrc;
	};



}