set(SOURCES
    ArrayTypeTest.cpp
    IterationSpaceTest.cpp)

marco_add_unittest(ModelicaDialectTest ${SOURCES})

marco_link_llvm_libs(ModelicaDialectTest Core Support)

target_link_libraries(ModelicaDialectTest
    PRIVATE MLIRModelica)
