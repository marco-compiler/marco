#include "marco/Codegen/Lowering/BaseModelica/ScopedSymbolTable.h"

using namespace ::marco::codegen::lowering::bmodelica;

namespace marco::codegen::lowering::bmodelica {
ScopedSymbolTable::Scope::Scope(ScopedSymbolTable &table)
    : table(table), baseScope(table.table), size(table.names.size()) {}

ScopedSymbolTable::Scope::~Scope() { table.names.resize(size); }

void ScopedSymbolTable::insert(llvm::StringRef name, Reference reference,
                               SymbolType type) {
  table.insert(name, {reference, type});
  names.push_back(name.str());
}

std::optional<SymbolInfo>
ScopedSymbolTable::lookup(llvm::StringRef name) const {
  if (table.count(name) == 0) {
    return std::nullopt;
  }

  return table.lookup(name);
}

llvm::StringSet<> ScopedSymbolTable::getSymbolNames(
    llvm::function_ref<bool(llvm::StringRef, Reference, SymbolType)> filterFn)
    const {
  llvm::StringSet<> result;

  for (const std::string &name : names) {
    SymbolInfo symbolInfo = table.lookup(name);

    if (filterFn(name, symbolInfo.reference, symbolInfo.type)) {
      result.insert(name);
    }
  }

  return result;
}
} // namespace marco::codegen::lowering::bmodelica
