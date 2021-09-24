set(SOURCES
        src/dialects/ida/Attribute.cpp
        src/dialects/ida/IdaBuilder.cpp
        src/dialects/ida/IdaDialect.cpp
        src/dialects/ida/Ops.cpp
        src/dialects/ida/Type.cpp
        src/dialects/modelica/Attribute.cpp
        src/dialects/modelica/ModelicaBuilder.cpp
        src/dialects/modelica/ModelicaDialect.cpp
        src/dialects/modelica/Ops.cpp
        src/dialects/modelica/Type.cpp
        src/passes/matching/Edge.cpp
        src/passes/matching/Flow.cpp
        src/passes/matching/LinSolver.cpp
        src/passes/matching/MatchedEquationLookup.cpp
        src/passes/matching/Matching.cpp
        src/passes/matching/SCCCollapsing.cpp
        src/passes/matching/SCCDependencyGraph.cpp
        src/passes/matching/Schedule.cpp
        src/passes/matching/SVarDependencyGraph.cpp
        src/passes/matching/VVarDependencyGraph.cpp
        src/passes/model/BltBlock.cpp
        src/passes/model/Constant.cpp
        src/passes/model/Equation.cpp
        src/passes/model/Expression.cpp
        src/passes/model/Induction.cpp
        src/passes/model/Model.cpp
        src/passes/model/Path.cpp
        src/passes/model/Operation.cpp
        src/passes/model/Reference.cpp
        src/passes/model/ReferenceMatcher.cpp
        src/passes/model/Variable.cpp
        src/passes/model/VectorAccess.cpp
        src/passes/AutomaticDifferentiation.cpp
        src/passes/BufferDeallocation.cpp
        src/passes/ExplicitCastInsertion.cpp
        src/passes/FunctionsVectorization.cpp
        src/passes/IdaConversion.cpp
        src/passes/ModelicaConversion.cpp
        src/passes/LowerToCFG.cpp
        src/passes/LowerToLLVM.cpp
        src/passes/ResultBuffersToArgs.cpp
        src/passes/SolveModel.cpp
        src/passes/TypeConverter.cpp
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

marco_add_library(mlirlowerer SHARED ${SOURCES})

target_link_libraries(mlirlowerer
        PUBLIC marco::frontend
        PUBLIC marco::utils
        PUBLIC marco::runtime
        PRIVATE ${MLIR_LIBS}
        PRIVATE pthread
        PRIVATE boost)

add_compile_definitions(RUNTIME_LIBRARY="$<TARGET_FILE:runtime>")
