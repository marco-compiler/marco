#include "marco/Codegen/Lowering/VariablesSymbolTable.h"

namespace marco::codegen::lowering
{
  VariablesSymbolTable::VariablesScope::VariablesScope(VariablesSymbolTable &ht): 
      scope(ht.references) {};

  void VariablesSymbolTable::insert(const llvm::StringRef &Key, const Reference &Val) {
    references.insert(Key, Val);
    variablesSet.insert(Key.str());
  }

  unsigned int VariablesSymbolTable::count(const llvm::StringRef &Key) const {
    return references.count(Key);
  }

  std::optional<Reference> VariablesSymbolTable::lookup(const llvm::StringRef &Key) const {
    if (!count(Key)) {
      return std::nullopt;
    }
    return references.lookup(Key);
  }

  std::set<std::string> VariablesSymbolTable::getVariables(bool onlyVisible) const {
    std::set<std::string> result;

    for (auto pVar = variablesSet.cbegin(); pVar != variablesSet.cend(); ++pVar) {
      if (!onlyVisible || references.count(llvm::StringRef(*pVar))) {
        result.insert(*pVar);
      }
    }

    return result;
  }
}