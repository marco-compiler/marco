#ifndef MARCO_CODEGEN_LOWERING_LOWERINGCONTEXT_H
#define MARCO_CODEGEN_LOWERING_LOWERINGCONTEXT_H

#include "marco/Codegen/Lowering/Reference.h"
#include "marco/Codegen/Options.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

namespace marco::codegen::lowering
{
  struct LoweringContext
  {
    using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, Reference>;
    using SymbolScope = SymbolTable::ScopeTy;

    LoweringContext(mlir::MLIRContext& context, CodegenOptions options);

    /// The builder is a helper class to create IR inside a function. The
    /// builder is stateful, in particular it keeps an "insertion point",
    /// that is where the next operations will be introduced.
    mlir::OpBuilder builder;

    /// The symbol table maps a variable name to a value in the current scope.
    /// Entering a function creates a new scope, and the function arguments
    /// are added to the mapping. When the processing of a function is
    /// terminated, the scope is destroyed and the mappings created in this
    /// scope are dropped.
    SymbolTable symbolTable;

    /// A list of options that has impact on the whole code generation process.
    CodegenOptions options;
  };
}

#endif // MARCO_CODEGEN_LOWERING_LOWERINGCONTEXT_H
