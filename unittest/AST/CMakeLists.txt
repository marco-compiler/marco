set(SOURCES
    ArrayTest.cpp
    ExpressionTest.cpp
    StatementTest.cpp
    TupleTest.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ASTTest ${SOURCES})

target_link_libraries(ASTTest
    PRIVATE marco::ASTBaseModelica)
