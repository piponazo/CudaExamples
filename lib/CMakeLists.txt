add_library(utils
    include/lib/utils.h
    src/utils.cpp
)

add_executable(readtiff readtiff.cpp)

target_link_libraries(readtiff
    PRIVATE
        TIFF::TIFF
        ZLIB::ZLIB
)

target_link_libraries(utils
    PRIVATE
        JPEG::JPEG
        TIFF::TIFF
)

target_include_directories(utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
