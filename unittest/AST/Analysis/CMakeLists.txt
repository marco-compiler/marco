set(SOURCES
    DynamicDimensionsGraphTest.cpp)

marco_add_unittest(ASTAnalysisTest ${SOURCES})

marco_link_llvm_libs(ASTAnalysisTest Core Support)

target_link_libraries(ASTAnalysisTest
    PRIVATE marco::ASTAnalysis)
