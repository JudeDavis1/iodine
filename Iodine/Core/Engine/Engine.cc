#include "Engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <Python.h>
#include <vector>


using namespace Idn;
namespace py = pybind11;


namespace Idn {

Engine::Engine() {
    std::cout << "Initializing the interpreter..." << std::endl;
    Py_Initialize();
}

void Engine::Run() {
    py::module sys = py::module::import("sys");
    py::module pip = py::module::import("pip");
    py::function pip_install = pip.attr("main");

    std::string script = "import sysconfig; python_binary = sysconfig.get_config_var('BINDIR') + '/' + sysconfig.get_config_var('PYTHON')";
    std::string python_path = py::eval(script).cast<py::str>();

    system(python_path.c_str());
}

Engine::~Engine() {
    Py_Finalize();
}

}


