set(SOURCES
    marco.cpp
    marco-mc1.cpp)

marco_add_tool(marco-driver ${SOURCES})

target_link_libraries(marco-driver
    PRIVATE marco::frontend
    PRIVATE marco::frontendTool)

clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver)

install(TARGETS marco-driver RUNTIME DESTINATION bin)

# This shell script will only work in a POSIX shell.
if (NOT WIN32)
  add_custom_command(TARGET marco-driver POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
      ${CMAKE_CURRENT_SOURCE_DIR}/marco.sh
      $<TARGET_FILE_DIR:marco-driver>/marco)

  add_custom_target(copyRunScript ALL
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_SOURCE_DIR}/marco.sh
      $<TARGET_FILE_DIR:marco-driver>/marco)

  add_dependencies(copyRunScript marco-driver)

  install(PROGRAMS $<TARGET_FILE_DIR:marco-driver>/marco DESTINATION bin)
endif()
