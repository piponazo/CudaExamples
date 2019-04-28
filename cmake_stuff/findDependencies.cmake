# Check if the conan file exist to find the dependencies
if (EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    #conan_basic_setup(NO_OUTPUT_DIRS KEEP_RPATHS SKIP_STD TARGETS)
    conan_set_find_paths()

    message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
endif()

set (GFLAGS_USE_TARGET_NAMESPACE ON)
find_package(gflags 2.2.2 EXACT REQUIRED)
find_package(JPEG REQUIRED)
