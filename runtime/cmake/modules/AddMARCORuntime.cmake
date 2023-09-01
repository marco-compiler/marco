macro(marco_add_runtime_library name)
    add_library(${name} ${ARGN})
    add_dependencies(marco-runtime ${name})
    add_library(MARCORuntime::${name} ALIAS ${name})

    set_property(TARGET ${name} PROPERTY OUTPUT_NAME MARCORuntime${name})

    # Enable SUNDIALS
    if (MARCO_ENABLE_SUNDIALS)
        target_compile_definitions(${name} PRIVATE SUNDIALS_ENABLE)
    endif()

    install(TARGETS ${name}
            EXPORT MARCORuntimeTargets
            COMPONENT ${name}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    install(TARGETS ${name} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endmacro()

macro(marco_add_runtime_static_library name)
    marco_add_runtime_library(${name} STATIC ${ARGN})

    # Enable the profiling framework.
    if (MARCO_PROFILING)
        target_compile_definitions(${name} PUBLIC MARCO_PROFILING)
    endif()
endmacro()

macro(marco_add_runtime_shared_library name)
    marco_add_runtime_library(${name} SHARED ${ARGN})
endmacro()

function(marco_add_unittest test_name)
    set(test_suite MARCORuntimeUnitTests)
    add_executable(${test_name} ${ARGN})

    target_link_directories(${test_name} PRIVATE ${MARCO_LIBS_DIR})
    target_link_libraries(${test_name} PRIVATE gtest_main gmock)

    add_dependencies(${test_suite} ${test_name})
    get_target_property(test_suite_folder ${test_suite} FOLDER)

    if (test_suite_folder)
        set_property(TARGET ${test_name} PROPERTY FOLDER "${test_suite_folder}")
    endif()

    gtest_discover_tests(${test_name})
endfunction()
