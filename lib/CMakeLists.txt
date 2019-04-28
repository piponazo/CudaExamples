add_library(utils
    include/lib/utils.h
    src/utils.cpp
)

target_link_libraries(utils PRIVATE JPEG::JPEG)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
