include(FindDoxygen)

if(DOXYGEN_FOUND)
  set(abs_top_srcdir ${CMAKE_CURRENT_SOURCE_DIR}/..)
  set(abs_top_builddir ..)

  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

  add_custom_target(doxygen-marco
      COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating MARCO doxygen documentation." VERBATIM)

  if (MARCO_BUILD_DOCS)
    add_dependencies(doxygen doxygen-marco)
  endif()

  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html DESTINATION docs)
endif()
