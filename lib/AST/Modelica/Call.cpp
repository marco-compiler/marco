#include "marco/AST/Modelica/Call.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void Call::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(callee, callback);
  walk(arguments, callback);
  walk(namedArguments, callback);
  walk(iterators, callback);
}

void Call::addJSONProperties(llvm::json::Object &obj) const {
  obj["callee"] = modelica::toJSON(callee.get());
  obj["arguments"] = modelica::toJSON(arguments);

  if (!namedArguments.empty()) {
    obj["named_arguments"] = modelica::toJSON(namedArguments);
  }

  if (!iterators.empty()) {
    obj["iterators"] = modelica::toJSON(iterators);
  }
}

ComponentReference *Call::getCallee() const { return callee.get(); }

void Call::setCallee(std::unique_ptr<ComponentReference> node) {
  callee = std::move(node);
}

ChildrenRef<Expression> Call::getArguments() const { return arguments; }

void Call::addArgument(std::unique_ptr<Expression> node) {
  arguments.emplace_back(std::move(node));
}

ChildrenRef<NamedArgument> Call::getNamedArguments() const {
  return namedArguments;
}

ChildrenRef<ForIndex> Call::getIterators() const { return iterators; }

void Call::addNamedArgument(std::unique_ptr<NamedArgument> node) {
  namedArguments.emplace_back(std::move(node));
}

void Call::addIterator(std::unique_ptr<ForIndex> node) {
  iterators.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
