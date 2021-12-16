set(SOURCES
        src/dialects/modelica/Attribute.cpp
        src/dialects/modelica/ModelicaBuilder.cpp
        src/dialects/modelica/ModelicaDialect.cpp
        src/dialects/modelica/Ops.cpp
        src/dialects/modelica/Type.cpp
        src/passes/AutomaticDifferentiation.cpp
        src/passes/BufferDeallocation.cpp
        src/passes/ExplicitCastInsertion.cpp
        src/passes/FunctionsVectorization.cpp
        src/passes/ModelicaConversion.cpp
        src/passes/LowerToCFG.cpp
        src/passes/LowerToLLVM.cpp
        src/passes/ResultBuffersToArgs.cpp
        src/passes/SolveModel.cpp
        src/passes/TypeConverter.cpp
        src/passes/model/Equation.cpp
        src/passes/model/Path.cpp
        src/passes/model/Variable.cpp
        src/passes/model/VVarDependencyGraph.cpp
        src/CodeGen.cpp
        src/Runner.cpp)

set(MLIR_LIBS
        MLIRIR
        MLIRMathToLLVM
        MLIROpenMPToLLVM
        MLIROpenMPToLLVMIRTranslation
        MLIRSCF
        MLIRSCFToStandard
        MLIRSCFToOpenMP
        MLIRStandard
        MLIRStandardToLLVM
        MLIRTransforms
        MLIRTranslation
        MLIRExecutionEngine)

marco_add_library(codegen SHARED ${SOURCES})

target_link_libraries(codegen
        PUBLIC marco::frontend
        PUBLIC marco::matching
        PUBLIC marco::runtime
        PUBLIC marco::utils
        PRIVATE ${MLIR_LIBS}
        PRIVATE pthread
        PRIVATE boost)

add_subdirectory(test)

add_compile_definitions(RUNTIME_LIBRARY="$<TARGET_FILE:runtime>")
