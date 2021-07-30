cmake_minimum_required(VERSION 3.13)
project(iwyu_base_case)

set(CMAKE_CXX_STANDARD 20)

add_executable(iwyu_base_case main.cpp)
set_property(TARGET iwyu_base_case PROPERTY CXX_INCLUDE_WHAT_YOU_USE "/usr/local/bin/include-what-you-use")