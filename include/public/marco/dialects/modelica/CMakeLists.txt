add_mlir_dialect(Modelica modelica)

#set(LLVM_TARGET_DEFINITIONS Modelica.td)
#mlir_tablegen(ModelicaAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=modelica)
#mlir_tablegen(ModelicaAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=modelica)
#add_public_tablegen_target(MLIRModelicaAttributesIncGen)
#add_dependencies(MLIRModelicaIncGen MLIRModelicaAttributesIncGen)
