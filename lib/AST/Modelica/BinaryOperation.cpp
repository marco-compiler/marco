#include "marco/AST/Modelica/BinaryOperation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void BinaryOperation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(lhs, callback);
  walk(rhs, callback);
}

void BinaryOperation::addJSONProperties(llvm::json::Object &obj) const {
  obj["operator"] = toString(op);
  obj["lhs"] = modelica::toJSON(lhs.get());
  obj["rhs"] = modelica::toJSON(rhs.get());
}

BinaryOperator BinaryOperation::getOperator() const { return op; }

void BinaryOperation::setOperator(BinaryOperator newOp) { op = newOp; }

Expression *BinaryOperation::getLhs() const { return lhs.get(); }

void BinaryOperation::setLhs(std::unique_ptr<Expression> node) {
  lhs = std::move(node);
}

Expression *BinaryOperation::getRhs() const { return rhs.get(); }

void BinaryOperation::setRhs(std::unique_ptr<Expression> node) {
  rhs = std::move(node);
}

llvm::StringRef toString(BinaryOperator op) {
  switch (op) {
  case BinaryOperator::Add:
    return "+";
    
  case BinaryOperator::AddEW:
    return ".+";

  case BinaryOperator::Sub:
    return "-";

  case BinaryOperator::SubEW:
    return ".-";

  case BinaryOperator::Mul:
    return "*";

  case BinaryOperator::MulEW:
    return ".*";

  case BinaryOperator::Div:
    return "/";

  case BinaryOperator::DivEW:
    return "./";

  case BinaryOperator::Pow:
    return "^";

  case BinaryOperator::PowEW:
    return ".^";
    
  case BinaryOperator::Equal:
    return "==";

  case BinaryOperator::NotEqual:
    return "<>";

  case BinaryOperator::Less:
    return "<";

  case BinaryOperator::LessEqual:
    return "<=";

  case BinaryOperator::Greater:
    return ">";

  case BinaryOperator::GreaterEqual:
    return ">=";

  case BinaryOperator::And:
    return "and";

  case BinaryOperator::Or:
    return "or";
  }

  llvm_unreachable("Unknown binary operator");
  return "";
}
} // namespace marco::ast::modelica
