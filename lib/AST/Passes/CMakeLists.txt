set(SOURCES
    ConstantFoldingPass.cpp
    InliningPass.cpp
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

    LINK_COMPONENTS
    Core
    Support)
