#include "Shader.h"

#include <sstream>
#include <fstream>



namespace Idn
{
	Shader::Shader(std::string vtx_path, std::string frag_path)
	{
		std::ifstream vtx_file, frag_file;
		std::stringstream vtx_stream, frag_stream;

		vtx_file.exceptions(std::ifstream::badbit);
		frag_file.exceptions(std::ifstream::badbit);

		try
		{
			// Load files
			vtx_file.open(vtx_path);
			frag_file.open(frag_path);
			
			// Store file contents in string streams

			// Copy the stream strings to string member vars
			m_vtxSrc = std::string((std::istreambuf_iterator<char>(vtx_file)), (std::istreambuf_iterator<char>()));
			m_fragSrc = std::string((std::istreambuf_iterator<char>(frag_file)), (std::istreambuf_iterator<char>()));
		}
		catch (std::ifstream::failure e)
		{
			std::cout << "Could not read file!" << std::endl;
		}
	}

	bool Shader::Compile()
	{
		const char* vs_cstr = m_vtxSrc.c_str();
		const char* fs_cstr = m_fragSrc.c_str();

		GLint success;
		GLuint vs, fs;
		char info_log[512];

		// Create and compile vertex shader
		vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vs_cstr, NULL);
		glCompileShader(vs);
		glGetShaderiv(vs, GL_COMPILE_STATUS, &success);

		if (!success)
		{
			glGetShaderInfoLog(vs, 512, NULL, info_log);
			std::cout << "Vertex shader compilation failed: " << info_log << std::endl;
		}

		// Create and compile fragment shader
		fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fs_cstr, NULL);
		glCompileShader(fs);
		glGetShaderiv(fs, GL_COMPILE_STATUS, &success);

		if (!success)
		{
			glGetShaderInfoLog(fs, 512, NULL, info_log);
			std::cout << "Fragment shader compilation failed: " << info_log << std::endl;
		}

		// Initialize a program from vertex and fragment shader
		m_program = glCreateProgram();

		glAttachShader(m_program, vs);
		glAttachShader(m_program, fs);
		glLinkProgram(m_program);

		// Get linker status
		glGetProgramiv(m_program, GL_LINK_STATUS, &success);

		if (!success)
		{
			glGetProgramInfoLog(m_program, 512, NULL, info_log);
			std::cout << "Failed to link program: " << info_log << std::endl;
		}

		glDeleteShader(vs);
		glDeleteShader(fs);

		return success;
	}

	void Shader::Use()
	{
		if (m_program == NULL)
		{
			std::cout << "Program is NULL! Please try Shader::Compile() before using it." << std::endl;
			return;
		}
		glUseProgram(m_program);
	}

	Shader::~Shader()
	{

	}
}


