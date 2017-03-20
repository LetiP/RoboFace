#! /bin/bash
dir=$(pwd)
pylibs="-lpython2.7 -lboost_python -lboost_system "
rapalib="-Wl,-whole-archive -lRapaPololuMaestro -Wl,-no-whole-archive"

# compile c++ tests
g++ -std=c++11 $dir/src/emotions.cxx -I $dir/include/ -o $dir/bin/emotions -lRapaPololuMaestro
g++ -std=c++11 $dir/src/servos.cxx -I $dir/include/ -o $dir/bin/servos -lRapaPololuMaestro

# compile python modules
g++ -std=c++11 $dir/python_bindings/face.cxx -o $dir/lib/face.o -c -fpic -I /usr/include/python2.7 -I $dir/include
g++ -std=c++11 -shared $dir/lib/face.o -o $dir/lib/face.so -lpython2.7 $pylibs $rapalib
