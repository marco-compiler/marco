if (MARCO_BUILD_DOXYGEN_DOCS)
  include(FindDoxygen)

  if (DOXYGEN_FOUND)
    set(abs_top_srcdir ${CMAKE_CURRENT_SOURCE_DIR}/..)
    set(abs_top_builddir ..)

    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/doxygen.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg @ONLY)

    add_custom_target(doxygen
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxygen.cfg
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating MARCO doxygen documentation." VERBATIM)

    add_dependencies(docs doxygen)

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html/
            DESTINATION docs/Doxygen)
  endif()
endif()

if (MARCO_BUILD_MLIR_DOCS)
  add_dependencies(docs mlir-doc)

  install(DIRECTORY ${CMAKE_BINARY_DIR}/docs/Dialects
          DESTINATION docs/Markdown)
endif()
