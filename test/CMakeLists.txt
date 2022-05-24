configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

# Working, but due to a bug within lit some tests would be marked as unresolved.
# See https://reviews.llvm.org/D100043
# For now we keep it disabled but still retaining the code, maybe a future fix will
# allow for tests unification. The python scripts are also renamed, in order to
# avoid the lit discovery process to find them.

#configure_lit_site_cfg(
#    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.py.in
#    ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg.py
#    MAIN_CONFIG
#    ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.cfg.py
#)

set(MARCO_TEST_DEPENDS
    FileCheck count not
    mlir-cpu-runner count not
    marco-driver
    modelica-opt
    marco::runtimeShared)

list(APPEND MARCO_TEST_DEPENDS MARCOUnitTests)

add_lit_testsuite(check "Running the regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${MARCO_TEST_DEPENDS})

set_target_properties(check PROPERTIES FOLDER "Tests")

add_lit_testsuites(MARCO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${MARCO_TEST_DEPENDS})
