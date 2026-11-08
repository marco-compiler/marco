set(SOURCES
    Lowerer.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(LoweringModelicaTest ${SOURCES})

target_link_libraries(LoweringModelicaTest
    PRIVATE
    marco::LoweringModelica
    marco::ParserModelica)
