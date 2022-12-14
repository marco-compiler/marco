set(SOURCES
    ArrayTest.cpp
    ConstantFoldingTest.cpp
    ExpressionTest.cpp
    SemanticAnalysisTest.cpp
    StatementTest.cpp
    TupleTest.cpp
    TypeCheckingTest.cpp
    TypeInferenceTest.cpp)

marco_add_unittest(ASTTest ${SOURCES})

marco_link_llvm_libs(ASTTest Core Support)

target_link_libraries(ASTTest
    PRIVATE marco::ast
    PRIVATE marco::ASTAnalysis
    PRIVATE marco::ASTPasses)
