#include "marco/AST/Modelica/UnaryOperation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void UnaryOperation::walkChildren(llvm::function_ref<void(Node *)> callback) {
  walk(operand, callback);
}

void UnaryOperation::addJSONProperties(llvm::json::Object &obj) const {
  obj["operator"] = toString(op);
  obj["operand"] = modelica::toJSON(operand.get());
}

UnaryOperator UnaryOperation::getOperator() const { return op; }

void UnaryOperation::setOperator(UnaryOperator newOp) { op = newOp; }

Expression *UnaryOperation::getOperand() const { return operand.get(); }

void UnaryOperation::setOperand(std::unique_ptr<Expression> node) {
  operand = std::move(node);
}

llvm::StringRef toString(UnaryOperator op) {
  switch (op) {
  case UnaryOperator::Plus:
    return "+";
  case UnaryOperator::Minus:
    return "-";
  case UnaryOperator::PlusEW:
    return ".+";
  case UnaryOperator::MinusEW:
    return ".-";
  case UnaryOperator::Not:
    return "not";
  }

  llvm_unreachable("Unknown unary operator");
  return "";
}
} // namespace marco::ast::modelica
