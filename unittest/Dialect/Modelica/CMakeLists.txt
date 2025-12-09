set(SOURCES
    ArrayTypeTest.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ModelicaDialectTest ${SOURCES})

target_link_libraries(ModelicaDialectTest
    PRIVATE MLIRBaseModelica)
