add_mlir_doc(BaseModelica BaseModelica Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS BaseModelicaAttributes.td)
mlir_tablegen(BaseModelicaAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(BaseModelicaAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRBaseModelicaAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS BaseModelicaTypes.td)
mlir_tablegen(BaseModelicaTypes.h.inc -gen-typedef-decls -typedefs-dialect=bmodelica)
mlir_tablegen(BaseModelicaTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=bmodelica)
add_public_tablegen_target(MLIRBaseModelicaTypesIncGen)

set(LLVM_TARGET_DEFINITIONS BaseModelicaOps.td)
mlir_tablegen(BaseModelicaOps.h.inc -gen-op-decls)
mlir_tablegen(BaseModelicaOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRBaseModelicaOpsIncGen)

set(LLVM_TARGET_DEFINITIONS BaseModelicaAttrInterfaces.td)
mlir_tablegen(BaseModelicaAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(BaseModelicaAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRBaseModelicaAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS BaseModelicaTypeInterfaces.td)
mlir_tablegen(BaseModelicaTypeInterfaces.h.inc -gen-type-interface-decls)
mlir_tablegen(BaseModelicaTypeInterfaces.cpp.inc -gen-type-interface-defs)
add_public_tablegen_target(MLIRBaseModelicaTypeInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS BaseModelicaOpInterfaces.td)
mlir_tablegen(BaseModelicaOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(BaseModelicaOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRBaseModelicaOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS BaseModelica.td)
mlir_tablegen(BaseModelica.h.inc -gen-dialect-decls -dialect=bmodelica)
mlir_tablegen(BaseModelica.cpp.inc -gen-dialect-defs -dialect=bmodelica)
add_public_tablegen_target(MLIRBaseModelicaIncGen)

add_dependencies(MLIRBaseModelicaIncGen MLIRBaseModelicaAttributesIncGen)
add_dependencies(MLIRBaseModelicaIncGen MLIRBaseModelicaTypesIncGen)
add_dependencies(MLIRBaseModelicaIncGen MLIRBaseModelicaOpsIncGen)
add_dependencies(MLIRBaseModelicaIncGen MLIRBaseModelicaOpInterfacesIncGen)
add_dependencies(MLIRBaseModelicaIncGen MLIRBaseModelicaAttrInterfacesIncGen)
add_dependencies(MLIRBaseModelicaIncGen MLIRBaseModelicaTypeInterfacesIncGen)

add_mlir_doc(BaseModelicaAttributes BaseModelicaAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(BaseModelicaTypes BaseModelicaTypes Dialects/ -gen-typedef-doc)
add_mlir_doc(BaseModelicaOps BaseModelicaOps Dialects/ -gen-op-doc)
