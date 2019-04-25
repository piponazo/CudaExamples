set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror=return-type")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden") # All the symbols will be hidden by default.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wreturn-type")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-long-long")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")

# ==========================================================
# CUDA stuff
# ==========================================================
#set(CUDA_NVCC_FLAGS "-gencode=arch=compute_20,code=sm_21")
#set(NVCC_DEBUG_FLAGS "-G")
#set(NVCC_RELEASE_FLAGS "-O3")

#we propagate manually
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

#if(UNIX)
#		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
#endif()

#if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
#		set(HOST_CXX_FLAGS "${HOST_CXX_FLAGS} ${CMAKE_CXX_FLAGS_DEBUG}")
#		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_DEBUG_FLAGS}")
#elseif(${CMAKE_BUILD_TYPE} STREQUAL "Release")
#		set(HOST_CXX_FLAGS "${HOST_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE}")
#		set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_RELEASE_FLAGS}")
#endif()
