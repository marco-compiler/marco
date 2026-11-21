#include "marco/AST/Modelica/StringLiteral.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void StringLiteral::walkChildren(llvm::function_ref<void(Node *)>) {}

void StringLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;
}

llvm::StringRef StringLiteral::getValue() const { return value; }

void StringLiteral::setValue(std::string newValue) {
  value = std::move(newValue);
}
} // namespace marco::ast::modelica
