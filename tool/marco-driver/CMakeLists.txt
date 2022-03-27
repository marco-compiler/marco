set(SOURCES
    marco.cpp
    marco-mc1.cpp)

marco_add_tool(marco-driver ${SOURCES})

target_link_libraries(marco-driver
    PRIVATE marco::frontend
    PRIVATE marco::frontendTool)

clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver
    LLVM-C) # needed to avoid undefined references

install(TARGETS marco-driver RUNTIME DESTINATION bin)

# This shell script will only work in a POSIX shell.
if (NOT WIN32)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/marco
      DESTINATION ${MARCO_TOOLS_DIR}
      FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)

  install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/marco DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
