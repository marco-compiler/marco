#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERINGCONTEXT_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERINGCONTEXT_H

#include "marco/Codegen/Lowering/BaseModelica/Reference.h"
#include "marco/Codegen/Lowering/BaseModelica/VariablesSymbolTable.h"
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

  mlir::SymbolTableCollection &getSymbolTable();

  VariablesSymbolTable &getVariablesSymbolTable();

  mlir::Operation *getLookupScope();

  void pushLookupScope(mlir::Operation *lookupScope);

private:
  /// The builder is a helper class to create IR inside a function. The
  /// builder is stateful, in particular it keeps an "insertion point",
  /// that is where the next operations will be introduced.
  mlir::OpBuilder builder;

  /// Global symbol table.
  /// It should not be used before the class declaration process is finished.
  mlir::SymbolTableCollection symbolTable;

  /// The symbol table maps a variable name to a value in the current scope.
  /// Entering a class creates a new scope. When the processing of a class is
  /// terminated, the scope is destroyed and the mappings created in this
  /// scope are dropped. However, the variable names can still be accesses after
  /// leaving the scope, to be used to provide debugging information to the
  /// user.
  VariablesSymbolTable variablesSymbolTable;

  llvm::SmallVector<mlir::Operation *> lookupScopes;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_LOWERINGCONTEXT_H
