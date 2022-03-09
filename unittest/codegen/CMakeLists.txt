set(SOURCES
    Matching.cpp)

marco_add_unittest(CodegenTest ${SOURCES})

marco_link_llvm_libs(CodegenTest Core Support)

target_link_libraries(CodegenTest
    PRIVATE marco::codegen)
