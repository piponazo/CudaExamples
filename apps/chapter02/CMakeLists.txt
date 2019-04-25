link_libraries(gflags::gflags)

add_executable(host_vectorAddition  host_vectorAddition.cpp)
add_executable(cuda_vectorAddition  cuda_vectorAddition.cu)

#target_link_libraries(host_vectorAddition PRIVATE gflags::gflags)

#cuda_add_executable(chapter03_book_sum_vector_cuda book_sum_vector.cu)
#cuda_add_executable(chapter03_exercise01 exercise01.cu)
#cuda_add_executable(chapter03_exercise02 exercise02.cu)
