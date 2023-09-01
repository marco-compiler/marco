set(SOURCES
    marco.cpp
    marco-mc1.cpp)

set(LLVM_LINK_COMPONENTS
    ${LLVM_TARGETS_TO_BUILD}
    Core
    MC
    MCParser
    Support)

marco_add_tool(marco ${SOURCES})

target_link_libraries(marco
    PRIVATE
    marco::frontend
    marco::driver)
