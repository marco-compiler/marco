#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERINGCONTEXT_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERINGCONTEXT_H

#include "marco/Codegen/Lowering/BaseModelica/Reference.h"
#include "marco/Codegen/Lowering/BaseModelica/ScopedSymbolTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace marco::codegen::lowering::bmodelica {
class LoweringContext {
public:
  class LookupScopeGuard {
  public:
    LookupScopeGuard(LoweringContext *context);

    ~LookupScopeGuard();

  private:
    LoweringContext *context;
    size_t size;
  };

  LoweringContext(mlir::MLIRContext &context);

  mlir::OpBuilder &getBuilder();

  mlir::SymbolTableCollection &getSymbolTables();

  ScopedSymbolTable &getScopedSymbolTable();

  const ScopedSymbolTable &getScopedSymbolTable() const;

  mlir::Operation *getLookupScope() const;

  void pushLookupScope(mlir::Operation *lookupScope);

private:
  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point",
  /// that is where the next operations will be introduced.
  mlir::OpBuilder builder;

  mlir::SymbolTableCollection symbolTables;
  ScopedSymbolTable scopedSymbolTable;
  llvm::SmallVector<mlir::Operation *> lookupScopes;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERINGCONTEXT_H
