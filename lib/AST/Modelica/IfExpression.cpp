#include "marco/AST/Modelica/IfExpression.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void IfExpression::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(conditions, callback);
  walk(values, callback);
  walk(elseValue, callback);
}

void IfExpression::addJSONProperties(llvm::json::Object &obj) const {
  obj["conditions"] = modelica::toJSON(conditions);
  obj["values"] = modelica::toJSON(values);
  obj["else"] = modelica::toJSON(elseValue.get());
}

ChildrenRef<Expression> IfExpression::getConditions() const {
  return conditions;
}

ChildrenRef<Expression> IfExpression::getValues() const { return values; }

void IfExpression::addBranch(std::unique_ptr<Expression> condition,
                             std::unique_ptr<Expression> value) {
  conditions.emplace_back(std::move(condition));
  values.emplace_back(std::move(value));
}

Expression *IfExpression::getElseValue() const { return elseValue.get(); }

void IfExpression::setElseValue(std::unique_ptr<Expression> node) {
  elseValue = std::move(node);
}
} // namespace marco::ast::modelica
