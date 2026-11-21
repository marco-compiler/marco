#include "marco/AST/Modelica/IntegerLiteral.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void IntegerLiteral::walkChildren(llvm::function_ref<void(Node *)>) {}

void IntegerLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;

  if (unit) {
    obj["unit"] = *unit;
  }
}

int64_t IntegerLiteral::getValue() const { return value; }

void IntegerLiteral::setValue(int64_t newValue) { value = newValue; }

const std::optional<std::string> &IntegerLiteral::getUnit() const {
  return unit;
}

void IntegerLiteral::setUnit(std::optional<std::string> newUnit) {
  unit = std::move(newUnit);
}
} // namespace marco::ast::modelica
