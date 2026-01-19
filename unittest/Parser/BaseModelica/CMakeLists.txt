set(SOURCES
    Parser.cpp
    StateMachine.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ParserBaseModelicaTest ${SOURCES})

target_link_libraries(ParserBaseModelicaTest
    PRIVATE marco::ParserBaseModelica)
