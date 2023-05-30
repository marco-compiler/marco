set(SOURCES
    BuiltInFunctionsTest.cpp
    MathTest.cpp
    UtilityFunctionsTest.cpp)

marco_add_unittest(RuntimeTest ${SOURCES})

marco_link_llvm_libs(RuntimeTest Core Support)

target_link_libraries(RuntimeTest
    PRIVATE marco::runtimeSupport)
