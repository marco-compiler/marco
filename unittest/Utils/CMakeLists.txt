set(SOURCES
    IRangeTest.cpp
    LexerTest.cpp
    NumbersLexerTest.cpp
    ScopeGuard.cpp
    VariableFilterTest.cpp)

marco_add_unittest(UtilsTest ${SOURCES})

marco_link_llvm_libs(UtilsTest Core Support)

target_link_libraries(UtilsTest
    PRIVATE marco::utils)
