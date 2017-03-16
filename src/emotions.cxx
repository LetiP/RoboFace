#include <iostream>
#include <string>
#include <face.h>

using namespace std;
using namespace face;

const string help_msg =
    "enter (0) for help\n"
    "enter (1) for neutral face\n"
    "enter (2) for unsure face\n"
    "enter (3) for happy face\n"
    "enter (4) for angry face\n"
    "enter (5) for sad face\n"
    "enter (6) to exit\n"
;


int main()
{
    Face f;

    cout << "Welcome user ...\n";
    cout << help_msg;

    for (int i = 0; i >= 0;)
    {
        cout << "enter number: ";
        cin >> i;

        switch(i)
        {
        case 0:
                cout << help_msg;
                break;
        case 1:
                cout << "neutral face\n\n";
                f.neutral();
                break;

        case 2:
                cout << "unsure face\n\n";
                f.unsure();
                break;

        case 3:
                cout << "happy face\n\n";
                f.happy();
                break;

        case 4:
                cout << "angry face\n\n";
                f.angry();
                break;

        case 5:
                cout << "sad face\n\n";
                f.sad();
                break;
        case 6:
                cout << "exiting...\n\n";
                i = -1;
                break;
        default:
                cout << "invalid input enter 0 for help\n\n";
                i = 0;
                break;
        }
    }
}
