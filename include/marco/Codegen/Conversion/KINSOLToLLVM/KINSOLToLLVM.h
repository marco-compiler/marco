#ifndef MARCO_CODEGEN_CONVERSION_KINSOLTOLLVM_KINSOLTOLLVM_H
#define MARCO_CODEGEN_CONVERSION_KINSOLTOLLVM_KINSOLTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/IR/DataLayout.h"

namespace mlir {
#define GEN_PASS_DECL_KINSOLTOLLVMCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"

namespace kinsol_to_llvm {
class GlobalsCache {
  llvm::DenseMap<mlir::Attribute, mlir::LLVM::GlobalOp> arrayConstants;

public:
  mlir::LLVM::GlobalOp
  getOrDeclareArrayConstant(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::SymbolTableCollection &symbolTables,
                            mlir::ModuleOp moduleOp,
                            mlir::DenseIntElementsAttr values);
};
} // namespace kinsol_to_llvm

void populateKINSOLToLLVMConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::LLVMTypeConverter &typeConverter,
    mlir::SymbolTableCollection &symbolTables,
    kinsol_to_llvm::GlobalsCache *globalsCache = nullptr);

std::unique_ptr<mlir::Pass> createKINSOLToLLVMConversionPass();
} // namespace mlir

#endif // MARCO_CODEGEN_CONVERSION_KINSOLTOLLVM_KINSOLTOLLVM_H
