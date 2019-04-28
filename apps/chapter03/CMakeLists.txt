link_libraries(gflags::gflags)

add_executable(cuda_color2grey  cuda_color2grey.cu)

target_link_libraries(cuda_color2grey PRIVATE utils)
