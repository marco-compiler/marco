#include "marco/AST/Modelica/EnumerationLiteral.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void EnumerationLiteral::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  description.walkChildren(callback);
}

void EnumerationLiteral::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;
  description.addToJSON(obj);
}

llvm::StringRef EnumerationLiteral::getName() const { return name; }

void EnumerationLiteral::setName(std::string newName) {
  name = std::move(newName);
}

Description &EnumerationLiteral::getDescription() { return description; }

const Description &EnumerationLiteral::getDescription() const {
  return description;
}
} // namespace marco::ast::modelica
