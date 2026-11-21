#include "marco/AST/Modelica/FunctionPartialApplication.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void FunctionPartialApplication::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(namedArguments, callback);
}

void FunctionPartialApplication::addJSONProperties(
    llvm::json::Object &obj) const {
  obj["function"] = modelica::toJSON(function);
  obj["named_arguments"] = modelica::toJSON(namedArguments);
}

const Name &FunctionPartialApplication::getFunction() const { return function; }

void FunctionPartialApplication::setFunction(Name name) {
  function = std::move(name);
}

ChildrenRef<NamedArgument>
FunctionPartialApplication::getNamedArguments() const {
  return namedArguments;
}

void FunctionPartialApplication::addNamedArgument(
    std::unique_ptr<NamedArgument> node) {
  namedArguments.emplace_back(std::move(node));
}
} // namespace marco::ast::modelica
