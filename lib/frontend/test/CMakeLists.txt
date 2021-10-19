set(SOURCES
        src/AlgorithmTest.cpp
        src/ArrayTest.cpp
        src/AnnotationTest.cpp
        src/ConstantFolderTest.cpp
        src/ExpressionTest.cpp
        src/FunctionTest.cpp
        src/LexerStateMachineTest.cpp
        src/ParserTest.cpp
        src/ParserEquationTest.cpp
        src/ParserClassTest.cpp
        src/RecordTest.cpp
        src/StatementTest.cpp
        src/TupleTest.cpp
        src/TypeCheckerTest.cpp)

marco_add_unittest(MARCOTest FrontendTest ${SOURCES})

marco_link_llvm_libs(FrontendTest core support)

target_link_libraries(FrontendTest
        PRIVATE marco::frontend)