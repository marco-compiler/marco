set(SOURCES
    marco.cpp
    marco-cc1.cpp
    marco-mc1.cpp)

marco_add_tool(marco-driver ${SOURCES})

target_link_libraries(marco-driver
    PRIVATE marco::frontend
    PRIVATE marco::frontendTool)

clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver)

install(TARGETS marco-driver RUNTIME DESTINATION bin)

# This flang shell script will only work in a POSIX shell.
if (NOT WIN32)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/marco.sh
      DESTINATION ${CMAKE_BINARY_DIR}/bin
      FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)

  file(RENAME ${CMAKE_BINARY_DIR}/bin/marco.sh ${CMAKE_BINARY_DIR}/bin/marco)

  install(PROGRAMS ${CMAKE_BINARY_DIR}/bin/marco DESTINATION bin)
endif()
