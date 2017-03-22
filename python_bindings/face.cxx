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
        .def("neutral", &Face::neutral, (arg("moveHead") = true),
                "neutral facial expression with optional head movement")
        .def("unsure", &Face::unsure, (arg("moveHead") = true),
                "unsure facial expression with optional head movement")
        .def("happy", &Face::happy, (arg("moveHead") = true),
                "happy facial expression with optional head movement")
        .def("angry", &Face::angry, (arg("moveHead") = true),
                "angry facial expression with optional head movement")
        .def("sad", &Face::sad, (arg("moveHead") = true),
                "sad facial expression with optional head movement")
        .def("moveHeadX", &Face::moveHeadX, (arg("x")),
                "move head horizontally")
        .def("moveHeadY", &Face::moveHeadY, (arg("y")),
                "move head vertically")
        .def("moveHead", &Face::moveHead, (arg("x"), arg("y")),
                "move head horizontally and vertically")
    ;
}
