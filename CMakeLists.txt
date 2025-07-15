cmake_minimum_required(VERSION 3.10)
project(LinearRegressionProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add directories with headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# List of src file
set(SOURCES
        main.cpp
        src/model/LinearRegression.cpp
        src/optimizer/GradientDescent.cpp
        src/utils/Loss.cpp
        src/utils/CSVReader.cpp
)

add_executable(linear_regression ${SOURCES})
