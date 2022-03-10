set(SOURCES
    dialects/modelica/Attribute.cpp
    dialects/modelica/ModelicaBuilder.cpp
    dialects/modelica/ModelicaDialect.cpp
    dialects/modelica/Ops.cpp
    dialects/modelica/Type.cpp
    Lowering/FunctionCallBridge.cpp
    Lowering/OperationBridge.cpp
    Lowering/Reference.cpp
    Lowering/Results.cpp
    Bridge.cpp
    NewBridge.cpp
    Options.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
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
    MLIRTranslation)

marco_add_library(codegen
    SHARED
    ${SOURCES}

    DEPENDS
    marco::ast
    marco::utils

    LINK_LIBS PUBLIC
    marco::ast
    marco::utils
    ${MLIR_LIBS})

add_subdirectory(Conversion)
add_subdirectory(Transforms)
