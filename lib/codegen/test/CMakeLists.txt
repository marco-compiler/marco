set(SOURCES
        src/Matching.cpp)

marco_add_unittest(MARCOTest CodegenTest ${SOURCES})

marco_link_llvm_libs(CodegenTest core support)

target_link_libraries(CodegenTest
        PRIVATE marco::codegen)