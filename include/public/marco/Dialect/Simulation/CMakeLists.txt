add_mlir_dialect(Simulation simulation)
add_mlir_doc(Simulation Simulation Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Simulation.td)
mlir_tablegen(SimulationAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(SimulationAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRSimulationAttributesIncGen)
add_dependencies(MLIRSimulationIncGen MLIRSimulationAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS SimulationInterfaces.td)
mlir_tablegen(SimulationInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(SimulationInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRSimulationInterfacesIncGen)
add_dependencies(MLIRSimulationIncGen MLIRSimulationInterfacesIncGen)
