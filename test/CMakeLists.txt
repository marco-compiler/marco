configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(MARCO_TEST_DEPENDS
        FileCheck count not
        mlir-cpu-runner count not
        marco
        modelica-opt
        marco::runtime)

add_lit_testsuite(check "Running the regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${MARCO_TEST_DEPENDS})

set_target_properties(check PROPERTIES FOLDER "Tests")

add_lit_testsuites(MARCO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MARCO_TEST_DEPENDS})
