#include "marco/AST/Modelica/RealLiteral.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void RealLiteral::walkChildren(llvm::function_ref<void(Node *)>) {}

void RealLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["value"] = value;

  if (unit) {
    obj["unit"] = *unit;
  }
}

double RealLiteral::getValue() const { return value; }

void RealLiteral::setValue(double newValue) { value = newValue; }

const std::optional<std::string> &RealLiteral::getUnit() const { return unit; }

void RealLiteral::setUnit(std::optional<std::string> newUnit) {
  unit = std::move(newUnit);
}
} // namespace marco::ast::modelica
