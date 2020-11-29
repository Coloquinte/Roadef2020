
cmake_minimum_required(VERSION 2.8)
PROJECT(ROADEF2020)

SET(CMAKE_CXX_FLAGS "-g -std=c++14")

#SET(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#SET(Boost_USE_STATIC_LIBS ON)
FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem iostreams program_options unit_test_framework)
INCLUDE_DIRECTORIES(${ROADEF2020_SOURCE_DIR}/include)

SET(SOURCES
  src/main.cpp
  src/problem.cpp
  src/bs_optimizer.cpp
  src/bs_analyzer.cpp
  src/measures.cpp
)

add_executable(roadef ${SOURCES})
target_link_libraries(roadef
  ${Boost_LIBRARIES}
)

