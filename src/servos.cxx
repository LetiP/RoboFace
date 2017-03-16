#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <face.h>

using namespace std;
using namespace face;


int main(int argc, char** argv)
{
    if (argc != 3)
    {
        cerr << "Usage: " << argv[0] << "<channelIdx> <servoPos>\n";
        return -1;
    }
    int channel = atoi(argv[1]);
    int servo = atoi(argv[2]);

    Face f;
    ServoConfig<1> config({channel}, {servo});
    f.applyConfig(config);

    return 0;
}
