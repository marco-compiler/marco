add_mlir_dialect_library(MLIRSimulation
    Attributes.cpp
    Interfaces.cpp
    SimulationDialect.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    MLIRSimulationIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRModeling)
