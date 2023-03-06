#include "Engine.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <Python.h>


namespace py = pybind11;
using namespace Idn;
using namespace py::literals;


namespace Idn {

Engine::Engine() {
    std::cout << "Initializing the interpreter..." << std::endl;
    py::scoped_interpreter guard {};
}

void Engine::InstallDeps() {
    py::module sys = py::module::import("sys");
    py::exec("import sysconfig; python_binary = sysconfig.get_config_var('BINDIR') + '/' + sysconfig.get_config_var('PYTHON')");
}

void Engine::Run() {
    this->InstallDeps();
}

Engine::~Engine() {
    Py_Finalize();
}

}


