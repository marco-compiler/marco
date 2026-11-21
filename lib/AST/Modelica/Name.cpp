#include "marco/AST/Modelica/Name.h"
#include "llvm/ADT/StringExtras.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
std::string Name::toString() const {
  std::string result = global ? "." : "";
  result += llvm::join(parts, ".");
  return result;
}

llvm::json::Value toJSON(const Name &name) { return name.toString(); }
} // namespace marco::ast::modelica
