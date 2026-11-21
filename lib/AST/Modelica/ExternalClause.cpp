#include "marco/AST/Modelica/ExternalClause.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ExternalClause::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(output, callback);
  walk(arguments, callback);
  walk(annotation, callback);
}

void ExternalClause::addJSONProperties(llvm::json::Object &obj) const {
  if (language) {
    obj["language"] = *language;
  }

  if (hasFunctionCall()) {
    obj["function"] = function;
    obj["arguments"] = modelica::toJSON(arguments);

    if (output) {
      obj["output"] = modelica::toJSON(output.get());
    }
  }

  if (annotation) {
    obj["annotation"] = modelica::toJSON(annotation.get());
  }
}

const std::optional<std::string> &ExternalClause::getLanguage() const {
  return language;
}

void ExternalClause::setLanguage(std::optional<std::string> value) {
  language = std::move(value);
}

bool ExternalClause::hasFunctionCall() const { return !function.empty(); }

llvm::StringRef ExternalClause::getFunction() const { return function; }

void ExternalClause::setFunction(std::string name) {
  function = std::move(name);
}

ComponentReference *ExternalClause::getOutput() const { return output.get(); }

void ExternalClause::setOutput(std::unique_ptr<ComponentReference> node) {
  output = std::move(node);
}

ChildrenRef<Expression> ExternalClause::getArguments() const {
  return arguments;
}

void ExternalClause::addArgument(std::unique_ptr<Expression> node) {
  arguments.emplace_back(std::move(node));
}

ClassModification *ExternalClause::getAnnotation() const {
  return annotation.get();
}

void ExternalClause::setAnnotation(std::unique_ptr<ClassModification> node) {
  annotation = std::move(node);
}
} // namespace marco::ast::modelica
