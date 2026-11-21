#include "marco/AST/Modelica/ClassModification.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ClassModification::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(arguments, callback);
}

void ClassModification::addJSONProperties(llvm::json::Object &obj) const {
  obj["arguments"] = modelica::toJSON(arguments);
}

ChildrenRef<Argument> ClassModification::getArguments() const {
  return arguments;
}

void ClassModification::addArgument(std::unique_ptr<Argument> node) {
  arguments.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
