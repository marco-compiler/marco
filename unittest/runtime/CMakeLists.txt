set(SOURCES
    ArrayDescriptorTest.cpp
    BuiltInFunctionsTest.cpp
    UtilityFunctionsTest.cpp)

marco_add_unittest(RuntimeTest ${SOURCES})

marco_link_llvm_libs(RuntimeTest Core Support)

target_link_libraries(RuntimeTest
    PUBLIC marco::runtimeShared)
