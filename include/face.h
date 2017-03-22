#ifndef MAESTRO_ROBOT_FACE_HXX
#define MAESTRO_ROBOT_FACE_HXX


#include <array>
#include <string>
#include <algorithm>
#include <utility>
#include <type_traits>
#include <stdexcept>
#include <RPMSerialInterface.h>


namespace face {


/*
 * The servo configuration describes the position of the servo-motors.
 * In case the configuration is incomplete the old values (aka servo positions)
 * are maintained. The class is rather basic consisting mostly of
 * getter- and setter-methods.
*/
template<size_t N>
class ServoConfig
{
public:
    ServoConfig(const std::array<int,N> & servoChannel, const std::array<int,N> & servoPos)
    : servoChannel_(servoChannel), servoPos_(servoPos)
    {
        if ((servoPos.size() != N) || (servoChannel.size() != N))
            std::invalid_argument("initializer list size must be equal to N");
    }

    size_t size() const { return servoPos_.size(); }

    int getPosition(int i) const { return servoPos_.at(i); }
    int getChannel(int i)  const { return servoChannel_.at(i); }

    const std::array<int, N> & getPositions() const { return servoPos_; }
    const std::array<int, N> & getChannels()  const { return servoChannel_; }

    void setPosition(int i, int pos)    { servoPos_.at(i)     = pos; }
    void setChannel(int i, int channel) { servoChannel_.at(i) = channel; }

    void setPositions(const std::array<int, N> & list)
    {
        if (list.size() != N)
            std::invalid_argument("initializer list size must be equal to N");
        servoPos_ = list;
    }
    void setChannels(const std::array<int, N> & list)
    {
        if (list.size() != N)
            std::invalid_argument("initializer list size must be equal to N");
        servoChannel_ = list;
    }

private:
    std::array<int, N> servoChannel_;
    std::array<int, N> servoPos_;
};


/*
 * The servo constraints are the min- and max-positions of each servo as well as a
 * "list" of all available servos. The face class (see below) only contains one static
 * member of this class. The end-user should not need to use this class explicitly.
 */
template<size_t NUM_SERVOS>
struct ServoConstraints
{
public:
    ServoConstraints(int min, int max, const std::array<int, NUM_SERVOS> & channels)
    : channels_(channels), minPos_(min), maxPos_(max)
    {
        if (channels_.size() != NUM_SERVOS)
            std::invalid_argument("initializer list size must be equal to NUM_SERVOS");
    }

    bool isValidChannel(int channel) const
    {
        if (std::find(channels_.begin(), channels_.end(), channel) == channels_.end())
            return false;
        return true;
    }

    bool isValidPosition(int pos) const
    {
        if (pos < minPos_ || maxPos_ < pos)
            return false;
        return true;
    }

    template<size_t N>
    typename std::enable_if<(N <= NUM_SERVOS), bool>::type
    isValidChannelArray(const std::array<int, N> & array) const
    {
        for (auto x : array)
            if (!isValidChannel(x))
                return false;
        return true;
    }

    template<size_t N>
    typename std::enable_if<(N <= NUM_SERVOS), bool>::type
    isValidPositionArray(const std::array<int, N> & array) const
    {
        for (auto x : array)
            if (!isValidPosition(x))
                return false;
        return true;
    }

    template<size_t N>
    typename std::enable_if<(N <= NUM_SERVOS), bool>::type
    isValidConfig(const ServoConfig<N> & config) const
    {
        return isValidChannelArray(config.getChannels()) || isValidPositionArray(config.getPositions());
    }

    int getMinPos() const { return minPos_; }
    int getMaxPos() const { return maxPos_; }
    int getRange() const { return maxPos_ - minPos_; }

private:
    std::array<int, NUM_SERVOS> channels_;
    int minPos_;
    int maxPos_;
};


/*
 * The class Face is the api which the end-user should use to communicate with the robot.
 * It is aware of the constraints which each servo configuration the user may apply to the
 * robot must respect. The api is based around the applyConfig function which issues move
 * commands to the robot. Every functionally like emotion display via the functions angry(),
 * happy(), etc. is build on top of the applyConfig function.
 */
#define NUMBER_OF_SERVOS 12

class Face
{
public:
    Face(int x_len = 640 , int y_len = 480, const std::string & dev = "/dev/ttyACM0")
    : x_len_(x_len), y_len_(y_len), serialInterface_(nullptr)
    {
        std::string error_msg;
        serialInterface_ = RPM::SerialInterface::createSerialInterface(dev, 9600, &error_msg);

        if (serialInterface_ == nullptr)
            throw std::runtime_error(error_msg);
    }

    ~Face()
    {
        delete serialInterface_;
    }

    void neutral(bool moveHead = true)
    {
        if (moveHead)
            unsafeApplyConfig(neutralFaceMoveHead);
        else
            unsafeApplyConfig(neutralFace);
    }

    void unsure(bool moveHead = true)
    {
        if (moveHead)
            unsafeApplyConfig(unsureFaceMoveHead);
        else
            unsafeApplyConfig(unsureFace);
    }

    void happy(bool moveHead = true)
    {
        if (moveHead)
            unsafeApplyConfig(happyFaceMoveHead);
        else
            unsafeApplyConfig(happyFace);
    }

    void angry(bool moveHead = true)
    {
        if (moveHead)
            unsafeApplyConfig(angryFaceMoveHead);
        else
            unsafeApplyConfig(angryFace);
    }

    void sad(bool moveHead = true)
    {
        if (moveHead)
            unsafeApplyConfig(sadFaceMoveHead);
        else
            unsafeApplyConfig(sadFace);
    }


    void moveHeadX(int x)
    {
        if (x > x_len_ || x < 0)
            throw std::out_of_range("the x coordinate must be in range [0, x_len]");

        ServoConfig<2> config = {{Face::headMoveServo1_, Face::headMoveServo2_},
                                 {mapX_servo1(x), mapX_servo2(x)}};
        applyConfig(config);
    }

    void moveHeadY(int y)
    {
        if (y > y_len_ || y < 0)
            throw std::out_of_range("the y coordinate must be in range [0, y_len]");

        ServoConfig<2> config = {{Face::headMoveServo1_, Face::headMoveServo2_},
                                 {mapY_servo1(y), mapY_servo2(y)}};
        applyConfig(config);
    }

    void moveHead(int x, int y)
    {
        if (x > x_len_ || x < 0)
            throw std::out_of_range("the x coordinate must be in range [0, x_len]");

        if (y > y_len_ || y < 0)
            throw std::out_of_range("the y coordinate must be in range [0, y_len]");

        ServoConfig<2> configX = {{Face::headMoveServo1_, Face::headMoveServo2_},
                                  {mapXY_servo1(x, y), mapXY_servo2(x, y)}};
        applyConfig(configX);
    }

    template<size_t N>
    void applyConfig(const ServoConfig<N> & config)
    {
        if (!constraints_.isValidConfig(config))
            throw std::invalid_argument("the given servo config does not obey the constraints of the face");

        for (auto i = 0; i < config.size(); ++i)
            serialInterface_->setTargetCP(config.getChannel(i), config.getPosition(i));
    }

    template<size_t N>
    void unsafeApplyConfig(const ServoConfig<N> & config)
    {
        for (auto i = 0; i < config.size(); ++i)
            serialInterface_->setTargetCP(config.getChannel(i), config.getPosition(i));
    }

    static size_t numServos() { return numServos_; }

    static const ServoConstraints<NUMBER_OF_SERVOS> & getConstraints() { return constraints_; }

    ServoConfig<NUMBER_OF_SERVOS> getConfig() const
    {
        ServoConfig<numServos_> config({0,1,2,3,4,5,6,7,8,9,10,12}, {0,0,0,0,0,0,0,0,0,0,0,0});
        for (auto i = 0; i < numServos_; ++i)
        {
            unsigned short pos; serialInterface_->getPositionCP(config.getChannel(i), pos);
            config.setPosition(config.getChannel(i), static_cast<int>(pos));
        }
        return std::move(config);
    }

private:
    int mapX_servo1(int x) const
    {
        int x_len_2 = x_len_ / 2;

        if (x < x_len_2)
            x = ((static_cast<float>(x) / x_len_2) * 2000) + 5000;
        else
            x = ((static_cast<float>(x - x_len_2) / x_len_2) * 1000) + 7000;

        return roundTo50(x);
    }

    int mapX_servo2(int x) const
    {
        x = ((static_cast<float>(x) / x_len_) * 1000) + 7000;
        return roundTo50(x);
    }

    int mapY_servo1(int y) const
    {
        y = ((static_cast<float>(y) / y_len_) * 1000) + 6000;
        return roundTo50(y);
    }

    int mapY_servo2(int y) const
    {
        y = ((static_cast<float>(y) / y_len_) * 200) + 7800;
        return roundTo5(y);
    }

    int mapXY_servo1(int x, int y) const
    {
        constexpr float weightX1 = 0.5;
        constexpr float weightY1 = 0.5;
        static_assert(weightX1 + weightY1 == 1.0, "weightX1 + weightY1 must be equal to 1");
        return weightX1 * mapX_servo1(x) + weightY1 * mapY_servo1(y);
    }

    int mapXY_servo2(int x, int y) const
    {
        constexpr float weightX2 = 0.5;
        constexpr float weightY2 = 0.5;
        static_assert(weightX2 + weightY2 == 1.0, "weightX2 + weightY2 must be equal to 1");
        return weightX2 * mapX_servo2(x) + weightY2 * mapY_servo2(y);
    }

    int roundTo50(int n) const
    {
        return 50 * (n / 50);
    }

    int roundTo5(int n) const
    {
        return 5 * (n / 5);
    }

    static constexpr size_t numServos_ = NUMBER_OF_SERVOS;
    static constexpr int headMoveServo1_= 0;
    static constexpr int headMoveServo2_ = 1;
    static const ServoConstraints<numServos_> constraints_;

    // basic emotions
    static const ServoConfig<12> neutralFaceMoveHead;
    static const ServoConfig<10> unsureFaceMoveHead;
    static const ServoConfig<10> happyFaceMoveHead;
    static const ServoConfig<10> angryFaceMoveHead;
    static const ServoConfig<12> sadFaceMoveHead;

    static const ServoConfig<10> neutralFace;
    static const ServoConfig<8> unsureFace;
    static const ServoConfig<8> happyFace;
    static const ServoConfig<8> angryFace;
    static const ServoConfig<10> sadFace;


    RPM::SerialInterface* serialInterface_;
    int x_len_;
    int y_len_;
};

#undef NUMBER_OF_SERVOS


const ServoConstraints<Face::numServos_> Face::constraints_ = {
    4000, 8000, {0,1,2,3,4,5,6,7,8,9,10,12}
};

const ServoConfig<12> Face::neutralFaceMoveHead = {
    {0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {6000, 8000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<10> Face::unsureFaceMoveHead = {
    {0,    1,    2,    3,    5,    6,    7,    8,    9,    10},
    {6000, 8000, 5000, 7000, 8000, 7000, 6250, 5000, 7000, 7000}
};

const ServoConfig<10> Face::happyFaceMoveHead = {
    {0,    1,    2,    3,    5,     6,    7,    8,    9,    10},
    {6000, 8000, 7000, 7000, 70000, 4000, 5000, 5000, 8000, 5500}
};

const ServoConfig<10> Face::angryFaceMoveHead = {
    {0,    1,    2,    3,    5,    6,    7,    8,    9,    10},
    {7000, 7800, 8000, 4800, 7750, 4800, 4500, 5750, 7500, 7200}
};

const ServoConfig<12> Face::sadFaceMoveHead = {
    {0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {7000, 7800, 4000, 4000, 6000, 4000, 4000, 8000, 8000, 8000, 8000, 6000}
};

const ServoConfig<10> Face::neutralFace = {
    {2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<8> Face::unsureFace = {
    {2,    3,    5,    6,    7,    8,    9,    10},
    {5000, 7000, 8000, 7000, 6250, 5000, 7000, 7000}
};

const ServoConfig<8> Face::happyFace = {
    {2,    3,    5,     6,    7,    8,    9,    10},
    {7000, 7000, 70000, 4000, 5000, 5000, 8000, 5500}
};

const ServoConfig<8> Face::angryFace = {
    {2,    3,    5,    6,    7,    8,    9,    10},
    {8000, 4800, 7750, 4800, 4500, 5750, 7500, 7200}
};

const ServoConfig<10> Face::sadFace = {
    {2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {4000, 4000, 6000, 4000, 4000, 8000, 8000, 8000, 8000, 6000}
};



} // end namespace face


#endif //  MAESTRO_ROBOT_FACE_HXX
