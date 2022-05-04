set(SOURCES
    ArrayDescriptorTest.cpp
    BuiltInFunctionsTest.cpp
    UtilityFunctionsTest.cpp)

marco_add_unittest(RuntimeTest ${SOURCES})

marco_link_llvm_libs(RuntimeTest Core Support)

target_link_libraries(RuntimeTest
    PRIVATE marco::runtimeShared)
if(MSVC)
    target_compile_options(RuntimeTest PRIVATE "/Oi-" "/Zc:preprocessor")
endif()