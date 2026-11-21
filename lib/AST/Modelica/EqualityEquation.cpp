#include "marco/AST/Modelica/EqualityEquation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void EqualityEquation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(lhs, callback);
  walk(rhs, callback);
  description.walkChildren(callback);
}

void EqualityEquation::addJSONProperties(llvm::json::Object &obj) const {
  obj["lhs"] = modelica::toJSON(lhs.get());
  obj["rhs"] = modelica::toJSON(rhs.get());
  Equation::addJSONProperties(obj);
}

Expression *EqualityEquation::getLhs() const { return lhs.get(); }

void EqualityEquation::setLhs(std::unique_ptr<Expression> node) {
  lhs = std::move(node);
}

Expression *EqualityEquation::getRhs() const { return rhs.get(); }

void EqualityEquation::setRhs(std::unique_ptr<Expression> node) {
  rhs = std::move(node);
}
} // namespace marco::ast::modelica
