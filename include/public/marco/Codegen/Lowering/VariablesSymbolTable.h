#ifndef MARCO_CODEGEN_LOWERING_VARIABLESSYMBOLTABLE_H
#define MARCO_CODEGEN_LOWERING_VARIABLESSYMBOLTABLE_H

#include "marco/Codegen/Lowering/Reference.h"
#include "llvm/ADT/ScopedHashTable.h"
#include <optional>
#include <set>
#include <string>

namespace marco::codegen::lowering
{
  class VariablesSymbolTable {
    public:
      class VariablesScope {
        public:
          explicit VariablesScope(VariablesSymbolTable &ht);
        private:
          llvm::ScopedHashTable<llvm::StringRef, Reference>::ScopeTy scope;
      };

      void insert(const llvm::StringRef &Key, const Reference &Val);

      unsigned int count(const llvm::StringRef &Key) const;

      std::optional<Reference> lookup(const llvm::StringRef &Key) const;

      // If "onlyVisible" is true, return all variables ever inserted into the
      // symbol table. Otherwise, only return current ones.
      std::set<std::string> getVariables(bool onlyVisible = true) const;

    private:
      llvm::ScopedHashTable<llvm::StringRef, Reference> references;
      std::set<std::string> variablesSet;
  };
}

#endif // MARCO_CODEGEN_LOWERING_VARIABLESSYMBOLTABLE_H