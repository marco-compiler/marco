#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_SYMBOL_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_SYMBOL_H

#include "marco/Codegen/Lowering/BaseModelica/Reference.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace marco::codegen::lowering::bmodelica {
enum class SymbolType { Variable, VariableBuiltIn, Other };

struct SymbolInfo {
  Reference reference;
  SymbolType type;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_SYMBOL_H
