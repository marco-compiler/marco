set(SOURCES
    marco.cpp
    marco-mc1.cpp)

marco_add_tool(marco-driver ${SOURCES})

target_link_libraries(marco-driver
    PRIVATE marco::frontend)

clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver)

install(TARGETS marco-driver RUNTIME DESTINATION bin)

# This shell script will only work in a POSIX shell.
if (NOT WIN32)
  set(SIMULATION_LIBS "sundials_ida sundials_nvecserial sundials_sunlinsolklu klu")

  # Build version of the driver script
  set(RUNTIME_LIBS_PATHS ${SUNDIALS_LIBRARY_DIR})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/marco ${CMAKE_CURRENT_BINARY_DIR}/build/marco @ONLY)

  file(COPY ${CMAKE_CURRENT_BINARY_DIR}/build/marco
      DESTINATION ${MARCO_TOOLS_DIR}
      FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)

  # Install version of the driver script
  set(RUNTIME_LIBS_PATHS "$SCRIPT_PATH/../dependencies/lib")
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/marco ${CMAKE_CURRENT_BINARY_DIR}/install/marco @ONLY)
  install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/install/marco DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
