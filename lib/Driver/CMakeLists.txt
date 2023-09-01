set(SOURCES
    Driver.cpp)

add_compile_definitions(
    LINKER=${MARCORuntime_LINKER}
    LIBS_PATH=${MARCORuntime_LIBS_PATH}
    DEPENDENCIES_LIBS_PATH=${MARCORuntime_DEPENDENCIES_LIB_PATHS}
    LINK_EXTRA_FLAGS=${MARCORuntime_EXTRA_LINK_ARGS})

marco_add_library(driver
    ${SOURCES}

    LINK_LIBS
    marco::IO
    marco::options

    LINK_COMPONENTS
    Option
    Support)
