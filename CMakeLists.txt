cmake_minimum_required(VERSION 3.12)
project(OpenCvDNN)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Find Qt6 package
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)

# Enable Qt's MOC (Meta Object Compiler)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Check if OpenCV was found
if(OpenCV_FOUND)
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# Check if Qt6 was found
if(Qt6_FOUND)
    message(STATUS "Qt6 version: ${Qt6_VERSION}")
    message(STATUS "Qt6 found at: ${Qt6_DIR}")
else()
    message(FATAL_ERROR "Qt6 not found!")
endif()

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(${PROJECT_NAME} 
    main.cpp
    opencvdnnwidget.cpp
    opencvdnnwidget.h
)

# Link OpenCV libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Link Qt6 libraries
target_link_libraries(${PROJECT_NAME} Qt6::Core Qt6::Widgets Qt6::Gui)

# Optional: Set compiler flags for better optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Optional: Enable additional compiler warnings
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")