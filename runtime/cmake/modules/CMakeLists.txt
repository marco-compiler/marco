include(CMakePackageConfigHelpers)

set(TARGETS_PATH "lib/cmake")

set(RUNTIME_LINKER ${CMAKE_CXX_COMPILER})

set(INCLUDE_PATH "include")
set(LIBS_PATH "lib")

list(APPEND EXTRA_LINK_ARGS
        ${CMAKE_THREAD_LIBS_INIT})

list(APPEND INSTALL_DEPENDENCIES_INCLUDE_PATHS
        dependencies/sundials/include)

list(APPEND INSTALL_DEPENDENCIES_LIB_PATHS
        dependencies/sundials/lib)

configure_package_config_file(
    MARCORuntimeConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/MARCORuntimeConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MARCORuntime"
    PATH_VARS
        TARGETS_PATH
        INCLUDE_PATH
        LIBS_PATH
        INSTALL_DEPENDENCIES_INCLUDE_PATHS
        INSTALL_DEPENDENCIES_LIB_PATHS)

install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/MARCORuntimeConfig.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/MARCORuntime")
