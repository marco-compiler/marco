set(SOURCES
    VariableFilterTest.cpp)

marco_add_unittest(VariableFilterTest ${SOURCES})

marco_link_llvm_libs(VariableFilterTest Core Support)

target_link_libraries(VariableFilterTest
    PRIVATE marco::variableFilter)
