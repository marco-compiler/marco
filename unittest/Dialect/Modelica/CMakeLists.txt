set(SOURCES
    ConnectionEquations.cpp
    ConnectionGraph.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(DialectModelicaTest ${SOURCES})

target_link_libraries(DialectModelicaTest
    PRIVATE
    MLIRModelicaTransforms)
