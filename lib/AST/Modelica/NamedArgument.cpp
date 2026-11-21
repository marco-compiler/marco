#include "marco/AST/Modelica/NamedArgument.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void NamedArgument::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(value, callback);
}

void NamedArgument::addJSONProperties(llvm::json::Object &obj) const {
  obj["name"] = name;
  obj["value"] = modelica::toJSON(value.get());
}

llvm::StringRef NamedArgument::getName() const { return name; }

void NamedArgument::setName(std::string newName) { name = std::move(newName); }

Expression *NamedArgument::getValue() const { return value.get(); }

void NamedArgument::setValue(std::unique_ptr<Expression> node) {
  value = std::move(node);
}
} // namespace marco::ast::modelica
