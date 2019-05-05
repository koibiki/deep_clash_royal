cmake_minimum_required(VERSION 3.12)
project(clash_royal_agent)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

add_library(c_opencv SHARED
        op_c.cpp
        agent.cpp
        detect/detect.cpp
        detect/finish_detect.cpp
        detect/running_detect.cpp
        detect/menu_detect.cpp
        detect/start_detect.cpp
        dnn/elixir_detect.cpp
        dnn/card_detect.cpp)
include_directories(./include)

target_link_libraries(c_opencv ${OpenCV_LIBS})

add_executable(clash_royal_agent main.cpp )

target_link_libraries(clash_royal_agent c_opencv)

