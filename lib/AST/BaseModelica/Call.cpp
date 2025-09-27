#include "marco/AST/BaseModelica/Call.h"
#include "marco/AST/BaseModelica/FunctionArgument.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
Call::Call(SourceRange location)
    : Expression(ASTNodeKind::Expression_Call, std::move(location)) {}

Call::Call(const Call &other) : Expression(other) {
  setCallee(other.callee->clone());
  setArguments(other.arguments);
}

Call::~Call() = default;

std::unique_ptr<ASTNode> Call::clone() const {
  return std::make_unique<Call>(*this);
}

llvm::json::Value Call::toJSON() const {
  llvm::json::Object result;
  result["callee"] = getCallee()->toJSON();

  llvm::SmallVector<llvm::json::Value> argsJson;

  for (const auto &arg : arguments) {
    argsJson.push_back(arg->toJSON());
  }

  result["args"] = llvm::json::Array(argsJson);

  addNodeKindToJSON(*this, result);
  return result;
}

bool Call::isLValue() const { return false; }

Expression *Call::getCallee() {
  assert(callee != nullptr && "Callee not set");
  return callee->cast<Expression>();
}

const Expression *Call::getCallee() const {
  assert(callee != nullptr && "Callee not set");
  return callee->cast<Expression>();
}

void Call::setCallee(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  callee = std::move(node);
  callee->setParent(this);
}

size_t Call::getNumOfArguments() const { return arguments.size(); }

FunctionArgument *Call::getArgument(size_t index) {
  assert(index < arguments.size());
  return arguments[index]->cast<FunctionArgument>();
}

const FunctionArgument *Call::getArgument(size_t index) const {
  assert(index < arguments.size());
  return arguments[index]->cast<FunctionArgument>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>> Call::getArguments() const {
  return arguments;
}

void Call::setArguments(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  arguments.clear();

  for (const auto &node : nodes) {
    assert(node->isa<FunctionArgument>());
    auto &clone = arguments.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
