set(SOURCES
    ConstantFoldingPass.cpp
    PassManager.cpp
    SemanticAnalysisPass.cpp
    StructuralConstantPropagationPass.cpp
    RaggedFlatteningPass.cpp
    TypeCheckingPass.cpp
    TypeInferencePass.cpp)

marco_add_library(ASTPasses
    ${SOURCES}

    LINK_LIBS
    PUBLIC marco::ast
    PUBLIC marco::ASTAnalysis

    LINK_COMPONENTS
    Core
    Support)
