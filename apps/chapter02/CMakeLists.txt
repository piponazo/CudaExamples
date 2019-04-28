link_libraries(gflags::gflags)

add_executable(host_vectorAddition  host_vectorAddition.cpp)
add_executable(cuda_vectorAddition  cuda_vectorAddition.cu)
