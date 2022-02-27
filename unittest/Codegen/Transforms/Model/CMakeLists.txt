set(SOURCES
    CyclesTest.cpp
    EquationTest.cpp
    MatchingTest.cpp
    SchedulingTest.cpp
    Utils.cpp
    VariableTest.cpp)

marco_add_unittest(ModelTransformTest ${SOURCES})

marco_link_llvm_libs(ModelTransformTest Core Support)

target_link_libraries(ModelTransformTest PRIVATE
    marco::codegen
    marco::modeling
    MLIRModelicaTransforms)
