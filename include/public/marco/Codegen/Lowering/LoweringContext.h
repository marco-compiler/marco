#ifndef MARCO_CODEGEN_LOWERING_LOWERINGCONTEXT_H
#define MARCO_CODEGEN_LOWERING_LOWERINGCONTEXT_H

#include <set>
#include <string>
#include "marco/Codegen/Lowering/Reference.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace marco::codegen::lowering
{
  class LoweringContext
  {
    public:
      using VariablesSymbolTable =
          llvm::ScopedHashTable<llvm::StringRef, Reference>;

      using VariablesScope = VariablesSymbolTable::ScopeTy;

      class LookupScopeGuard
      {
        public:
          LookupScopeGuard(LoweringContext* context);

          ~LookupScopeGuard();

        private:
          LoweringContext* context;
          size_t size;
      };

      LoweringContext(mlir::MLIRContext& context);

      mlir::OpBuilder& getBuilder();

      mlir::SymbolTableCollection& getSymbolTable();

      VariablesSymbolTable& getVariablesSymbolTable();

      std::set<llvm::StringRef>& getDeclaredVariables();

      mlir::Operation* getLookupScope();

      void pushLookupScope(mlir::Operation* lookupScope);

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
      /// scope are dropped.
      VariablesSymbolTable variablesSymbolTable;

      /// A set containing the variable names in "variablesSymbolTable". Used only 
      /// to print debugging information in case of parsing errors to the user.
      std::set<llvm::StringRef> declaredVariables;

      llvm::SmallVector<mlir::Operation*> lookupScopes;
  };
}

#endif // MARCO_CODEGEN_LOWERING_LOWERINGCONTEXT_H
