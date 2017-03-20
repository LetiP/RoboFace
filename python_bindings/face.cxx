#include <boost/python.hpp>
#include <face.h>


/* We only export a subset of the face functionality to python.
 * For more detail see include/face.h
 */
BOOST_PYTHON_MODULE_INIT(face)
{
    using namespace boost::python;
    using namespace face;

    class_<Face>("Face")
        .def("neutral", &Face::neutral)
        .def("unsure", &Face::unsure)
        .def("happy", &Face::happy)
        .def("angry", &Face::angry)
        .def("sad", &Face::sad)
        /*
        .def("moveHead", &Face::moveHead)
        .def("relativeMoveHead", &Face::relativeMoveHead)
        */
    ;
}
