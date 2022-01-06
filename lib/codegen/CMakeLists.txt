set(SOURCES
        dialects/modelica/Attribute.cpp
        dialects/modelica/ModelicaBuilder.cpp
        dialects/modelica/ModelicaDialect.cpp
        dialects/modelica/Ops.cpp
        dialects/modelica/Type.cpp
        passes/AutomaticDifferentiation.cpp
        passes/BufferDeallocation.cpp
        passes/ExplicitCastInsertion.cpp
        passes/FunctionsVectorization.cpp
        passes/ModelicaConversion.cpp
        passes/LowerToCFG.cpp
        passes/LowerToLLVM.cpp
        passes/ResultBuffersToArgs.cpp
        passes/SolveModel.cpp
        passes/TypeConverter.cpp
        passes/model/Equation.cpp
        passes/model/Path.cpp
        passes/model/Variable.cpp
        passes/model/VVarDependencyGraph.cpp
        CodeGen.cpp
        Runner.cpp)

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
        PUBLIC marco::ast
        PUBLIC marco::modeling
        PUBLIC marco::runtime
        PUBLIC marco::utils
        PRIVATE ${MLIR_LIBS}
        PRIVATE pthread
        PRIVATE boost)

add_compile_definitions(RUNTIME_LIBRARY="$<TARGET_FILE:runtime>")
