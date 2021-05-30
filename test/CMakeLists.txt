#configure_file(lit.site.cfg.py.in lit.site.cfg.py @ONLY)

#add_custom_target(check
#        COMMAND lit "${CMAKE_CURRENT_BINARY_DIR}" -v
#        DEPENDS modelica-opt)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(MARCO_TEST_DEPENDS
        FileCheck count not
        mlir-cpu-runner count not
        marco
        modelica-opt)

add_custom_target(check
        COMMAND lit "${CMAKE_CURRENT_BINARY_DIR}" -v
        DEPENDS ${MARCO_TEST_DEPENDS})
