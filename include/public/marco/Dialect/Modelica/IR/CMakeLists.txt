add_mlir_doc(Modelica Modelica Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS ModelicaAttributes.td)
mlir_tablegen(ModelicaAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(ModelicaAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRModelicaAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS ModelicaTypes.td)
mlir_tablegen(ModelicaTypes.h.inc -gen-typedef-decls -typedefs-dialect=modelica)
mlir_tablegen(ModelicaTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=modelica)
add_public_tablegen_target(MLIRModelicaTypesIncGen)

set(LLVM_TARGET_DEFINITIONS ModelicaOps.td)
mlir_tablegen(ModelicaOps.h.inc -gen-op-decls)
mlir_tablegen(ModelicaOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRModelicaOpsIncGen)

set(LLVM_TARGET_DEFINITIONS ModelicaAttrInterfaces.td)
mlir_tablegen(ModelicaAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(ModelicaAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRModelicaAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ModelicaTypeInterfaces.td)
mlir_tablegen(ModelicaTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(ModelicaTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRModelicaTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ModelicaOpInterfaces.td)
mlir_tablegen(ModelicaOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(ModelicaOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRModelicaOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Modelica.td)
mlir_tablegen(Modelica.h.inc -gen-dialect-decls -dialect=modelica)
mlir_tablegen(Modelica.cpp.inc -gen-dialect-defs -dialect=modelica)
add_public_tablegen_target(MLIRModelicaIncGen)

add_dependencies(MLIRModelicaIncGen MLIRModelicaAttributesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaTypesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaOpsIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaOpInterfacesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaAttrInterfacesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaTypeInterfacesIncGen)

add_mlir_doc(ModelicaAttributes ModelicaAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(ModelicaTypes ModelicaTypes Dialects/ -gen-typedef-doc)
add_mlir_doc(ModelicaOps ModelicaOps Dialects/ -gen-op-doc)
