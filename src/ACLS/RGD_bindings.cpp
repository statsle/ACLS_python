#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "RGD_core.h"


namespace py = pybind11;

PYBIND11_PLUGIN(RGD_bindings) {
    py::module m("RGD_bindings");
    py::options options;
    options.disable_function_signatures();
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("RGD", &acls::RGD, "A function which implements RGD");

    return m.ptr();
}