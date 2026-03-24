add_mlir_doc(Dialect Modelica Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Attributes.td)
mlir_tablegen(Attributes.h.inc -gen-attrdef-decls)
mlir_tablegen(Attributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRModelicaAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS Types.td)
mlir_tablegen(Types.h.inc -gen-typedef-decls -typedefs-dialect=modelica)
mlir_tablegen(Types.cpp.inc -gen-typedef-defs -typedefs-dialect=modelica)
add_public_tablegen_target(MLIRModelicaTypesIncGen)

set(LLVM_TARGET_DEFINITIONS Ops.td)
mlir_tablegen(Ops.h.inc -gen-op-decls)
mlir_tablegen(Ops.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRModelicaOpsIncGen)

set(LLVM_TARGET_DEFINITIONS AttrInterfaces.td)
mlir_tablegen(AttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(AttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRModelicaAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS TypeInterfaces.td)
mlir_tablegen(TypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(TypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRModelicaTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS OpInterfaces.td)
mlir_tablegen(OpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRModelicaOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS Dialect.td)
mlir_tablegen(Dialect.h.inc -gen-dialect-decls -dialect=modelica)
mlir_tablegen(Dialect.cpp.inc -gen-dialect-defs -dialect=modelica)
add_public_tablegen_target(MLIRModelicaIncGen)

add_dependencies(MLIRModelicaIncGen MLIRModelicaAttributesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaTypesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaOpsIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaOpInterfacesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaAttrInterfacesIncGen)
add_dependencies(MLIRModelicaIncGen MLIRModelicaTypeInterfacesIncGen)

add_mlir_doc(Attributes ModelicaAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(Types ModelicaTypes Dialects/ -gen-typedef-doc)
add_mlir_doc(Ops ModelicaOps Dialects/ -gen-op-doc)
