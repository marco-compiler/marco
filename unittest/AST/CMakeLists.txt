set(SOURCES
    AnnotationTest.cpp
    ArrayTest.cpp
    ConstantFolderTest.cpp
    ExpressionTest.cpp
    FunctionTest.cpp
    LexerStateMachineTest.cpp
    ParserClassTest.cpp
    ParserTest.cpp
    StatementTest.cpp
    TupleTest.cpp
    TypeCheckerTest.cpp)

marco_add_unittest(ASTTest ${SOURCES})

marco_link_llvm_libs(ASTTest Core Support)

target_link_libraries(ASTTest
    PRIVATE marco::ast
    PRIVATE marco::utils)
