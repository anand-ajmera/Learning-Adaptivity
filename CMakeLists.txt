cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(features_generator)

find_package(PCL 1.7 REQUIRED)

include_directories(
  ${PCL_INCLUDE_DIRS}
  include/
)

link_directories(
  ${PCL_LIBRARY_DIRS}
)

add_definitions(
  ${PCL_DEFINITIONS}
)

add_executable(feature_generator
  src/main.cpp
)

target_link_libraries(feature_generator
  ${PCL_LIBRARIES}
)
