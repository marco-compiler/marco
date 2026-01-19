set(SOURCES
    VariableFilter.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(VariableFilterTest ${SOURCES})

target_link_libraries(VariableFilterTest
    PRIVATE marco::variableFilter)
