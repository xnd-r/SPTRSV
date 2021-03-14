set(target ${LIBRARY})

file(GLOB srcs "*.cpp")
file(GLOB cls "*.cl")
file(GLOB hdrs "../include/*.h")

set_source_files_properties(${srcs} ${cls} ${hdrs} PROPERTIES
    LABELS "${MODULE};Library")

add_library(${target} STATIC ${srcs} ${cls} ${hdrs})
set_target_properties(${target} PROPERTIES
    OUTPUT_NAME ${MODULE}
    LABELS "${MODULE};Library")

target_compile_definitions(${target} PUBLIC ${BLA_DEFINITIONS})

if (UNIX)
  target_link_libraries(${target} ${CMAKE_THREAD_LIBS_INIT} PRIVATE ${BLAS_LIBRARIES} ${OPENCL_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
endif (UNIX)
target_link_libraries(${target} ${LIBRARY_DEPS})

set(LIBRARY_DEPS "${LIBRARY_DEPS};${target}" PARENT_SCOPE)
