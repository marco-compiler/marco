#include "marco/AST/Node/Operation.h"
#include "marco/AST/Node/Expression.h"
#include "llvm/ADT/StringRef.h"
#include <numeric>

using namespace ::marco;
using namespace ::marco::ast;

static std::string toString(OperationKind kind) {
  switch (kind) {
  case OperationKind::unknown:
    return "unknown";
  case OperationKind::negate:
    return "negate";
  case OperationKind::add:
    return "add";
  case OperationKind::addEW:
    return "add_ew";
  case OperationKind::subtract:
    return "subtract";
  case OperationKind::subtractEW:
    return "subtract_ew";
  case OperationKind::multiply:
    return "multiply";
  case OperationKind::multiplyEW:
    return "multiply_ew";
  case OperationKind::divide:
    return "divide";
  case OperationKind::divideEW:
    return "divide_ew";
  case OperationKind::ifelse:
    return "if_else";
  case OperationKind::greater:
    return "greater";
  case OperationKind::greaterEqual:
    return "greater_equal";
  case OperationKind::equal:
    return "equal";
  case OperationKind::different:
    return "different";
  case OperationKind::lessEqual:
    return "less_equal";
  case OperationKind::less:
    return "less";
  case OperationKind::land:
    return "land";
  case OperationKind::lor:
    return "lor";
  case OperationKind::lnot:
    return "lnot";
  case OperationKind::subscription:
    return "subscription";
  case OperationKind::powerOf:
    return "power_of";
  case OperationKind::powerOfEW:
    return "power_of_ew";
  case OperationKind::range:
    return "range";
  default:
    llvm_unreachable("Unknown operation kind");
    return "unknown";
  }
}

namespace marco::ast {
Operation::Operation(SourceRange location)
    : Expression(ASTNode::Kind::Expression_Operation, std::move(location)),
      kind(OperationKind::unknown) {}

Operation::Operation(const Operation &other)
    : Expression(other), kind(other.kind) {
  setArguments(other.arguments);
}

Operation::~Operation() = default;

std::unique_ptr<ASTNode> Operation::clone() const {
  return std::make_unique<Operation>(*this);
}

llvm::json::Value Operation::toJSON() const {
  llvm::json::Object result;
  result["operation_kind"] = toString(getOperationKind());

  llvm::SmallVector<llvm::json::Value> argsJson;

  for (const auto &arg : arguments) {
    argsJson.push_back(arg->toJSON());
  }

  result["args"] = llvm::json::Array(argsJson);

  addJSONProperties(result);
  return result;
}

bool Operation::isLValue() const {
  switch (getOperationKind()) {
  case OperationKind::subscription:
    return getArgument(0)->isLValue();

  default:
    return false;
  }
}

OperationKind Operation::getOperationKind() const { return kind; }

void Operation::setOperationKind(OperationKind newKind) {
  this->kind = newKind;
}

size_t Operation::getNumOfArguments() const { return arguments.size(); }

Expression *Operation::getArgument(size_t index) {
  assert(index < arguments.size());
  return arguments[index]->cast<Expression>();
}

const Expression *Operation::getArgument(size_t index) const {
  assert(index < arguments.size());
  return arguments[index]->cast<Expression>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Operation::getArguments() const {
  return arguments;
}

void Operation::setArguments(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  arguments.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = arguments.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
