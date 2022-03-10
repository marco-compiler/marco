add_mlir_dialect(Modelica modelica)
add_mlir_doc(Modelica Modelica Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Modelica.td)
mlir_tablegen(ModelicaAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ModelicaAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRModelicaAttributesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS ModelicaInterfaces.td)
mlir_tablegen(ModelicaInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ModelicaInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRModelicaInterfacesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaInterfacesIncGen)
