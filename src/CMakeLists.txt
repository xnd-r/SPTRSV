set(target_src ${LIBRARY})

file(GLOB srcs "*.cpp")
file(GLOB cls "*.cl")
file(GLOB hdrs "../include/*.h")

set_source_files_properties(${srcs} ${cls} ${hdrs} PROPERTIES
    LABELS "${MODULE};Library")

add_library(${target_src} STATIC ${srcs} ${cls} ${hdrs})
set_target_properties(${target_src} PROPERTIES
    OUTPUT_NAME ${MODULE}
    LABELS "${MODULE};Library")

target_compile_definitions(${target_src} PUBLIC ${BLA_DEFINITIONS})

if (UNIX)
  target_link_libraries(${target_src} ${CMAKE_THREAD_LIBS_INIT} PRIVATE ${BLAS_LIBRARIES} ${OPENCL_LIBRARIES} ${OpenMP_CXX_LIBRARIES})
endif (UNIX)
# target_link_libraries(${target_src} ${LIBRARY_DEPS})

# set(LIBRARY_DEPS "${LIBRARY_DEPS};${target_src}" PARENT_SCOPE)
