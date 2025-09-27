set(SOURCES
    ArrayTest.cpp
    ExpressionTest.cpp
    StatementTest.cpp
    TupleTest.cpp)

marco_add_unittest(ASTTest ${SOURCES})

marco_link_llvm_libs(ASTTest Core Support)

target_link_libraries(ASTTest
    PRIVATE marco::ASTBaseModelica)
