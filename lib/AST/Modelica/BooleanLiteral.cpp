#include "marco/AST/Modelica/BooleanLiteral.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void BooleanLiteral::walkChildren(llvm::function_ref<void(Node *)>) {}

void BooleanLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;
}

bool BooleanLiteral::getValue() const { return value; }

void BooleanLiteral::setValue(bool newValue) { value = newValue; }
} // namespace marco::ast::modelica
