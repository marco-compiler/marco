#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_SCOPEDSYMBOLTABLE_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_SCOPEDSYMBOLTABLE_H

#include "marco/Codegen/Lowering/BaseModelica/Symbol.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringSet.h"

namespace marco::codegen::lowering::bmodelica {
class ScopedSymbolTable {
  llvm::ScopedHashTable<llvm::StringRef, SymbolInfo> table;
  llvm::SmallVector<std::string> names;

public:
  class Scope {
    ScopedSymbolTable &table;
    llvm::ScopedHashTableScope<llvm::StringRef, SymbolInfo> baseScope;
    size_t size;

  public:
    explicit Scope(ScopedSymbolTable &table);

    ~Scope();
  };

public:
  void insert(llvm::StringRef name, Reference reference, SymbolType type);

  std::optional<SymbolInfo> lookup(llvm::StringRef name) const;

  llvm::StringSet<> getSymbolNames(
      llvm::function_ref<bool(llvm::StringRef, Reference, SymbolType)> filterFn)
      const;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_SCOPEDSYMBOLTABLE_H
