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
    Face() : serialInterface_(nullptr)
    {
        std::string error_msg;
        serialInterface_ = RPM::SerialInterface::createSerialInterface("/dev/ttyACM0", 9600, &error_msg);

        if (serialInterface_ == nullptr)
            throw std::runtime_error(error_msg);
    }

    ~Face()
    {
        delete serialInterface_;
    }

    void neutral() { unsafeApplyConfig(neutralFace); }
    void unsure()  { unsafeApplyConfig(unsureFace); }
    void happy()   { unsafeApplyConfig(happyFace); }
    void angry()   { unsafeApplyConfig(angryFace); }
    void sad()     { unsafeApplyConfig(sadFace); }

    void moveHead(int x, int y)
    {
        ServoConfig<2> config = {{Face::headMoveServoX_, Face::headMoveServoY_}, {x, y}};
        applyConfig(config);
    }

    void relativeMoveHead(int x_rel, int y_rel)
    {
        unsigned short x_abs; serialInterface_->getPositionCP(headMoveServoX_, x_abs);
        unsigned short y_abs; serialInterface_->getPositionCP(headMoveServoY_, y_abs);

        int x = static_cast<int>(x_abs) + x_rel;
        int y = static_cast<int>(y_abs) + y_rel;

        ServoConfig<2> config = {{headMoveServoX_, headMoveServoY_}, {x, y}};
        applyConfig(config);
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
    static constexpr size_t numServos_ = NUMBER_OF_SERVOS;
    static constexpr int headMoveServoX_= 4; // testing required
    static constexpr int headMoveServoY_ = 12; // testing required
    static const ServoConstraints<numServos_> constraints_;

    // basic emotions
    static const ServoConfig<12> neutralFace;
    static const ServoConfig<10> unsureFace;
    static const ServoConfig<10> happyFace;
    static const ServoConfig<10> angryFace;
    static const ServoConfig<12> sadFace;

    RPM::SerialInterface* serialInterface_;
};

#undef NUMBER_OF_SERVOS


const ServoConstraints<Face::numServos_> Face::constraints_ = {
    4000, 8000, {0,1,2,3,4,5,6,7,8,9,10,12}
};

const ServoConfig<12> Face::neutralFace = {
    {0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {6000, 8000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000}
};

const ServoConfig<10> Face::unsureFace = {
    {0,    1,    2,    3,    5,    6,    7,    8,    9,    10},
    {6000, 8000, 5000, 7000, 8000, 7000, 6250, 5000, 7000, 7000}
};

const ServoConfig<10> Face::happyFace = {
    {0,    1,    2,    3,    5,     6,    7,    8,    9,    10},
    {6000, 8000, 7000, 7000, 70000, 4000, 5000, 5000, 8000, 5500}
};

const ServoConfig<10> Face::angryFace = {
    {0,    1,    2,    3,    5,    6,    7,    8,    9,    10},
    {7000, 7800, 8000, 4800, 7750, 4800, 4500, 5750, 7500, 7200}
};

const ServoConfig<12> Face::sadFace = {
    {0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   12},
    {7000, 7800, 4000, 4000, 6000, 4000, 4000, 8000, 8000, 8000, 8000, 6000}
};


} // end namespace face


#endif //  MAESTRO_ROBOT_FACE_HXX
