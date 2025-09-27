#include "marco/AST/BaseModelica/Node/ExternalFunctionCall.h"
#include "marco/AST/BaseModelica/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast {
ExternalFunctionCall::ExternalFunctionCall(SourceRange location)
    : ASTNode(ASTNode::Kind::ExternalFunctionCall, std::move(location)) {}

ExternalFunctionCall::~ExternalFunctionCall() = default;

ExternalFunctionCall::ExternalFunctionCall(const ExternalFunctionCall &other)
    : ASTNode(other), callee(other.callee) {
  if (other.destination != nullptr) {
    setDestination(other.destination->clone());
  }

  setArguments(other.arguments);
}

std::unique_ptr<ASTNode> ExternalFunctionCall::clone() const {
  return std::make_unique<ExternalFunctionCall>(*this);
}

llvm::json::Value ExternalFunctionCall::toJSON() const {
  llvm::json::Object result;

  if (hasDestination()) {
    result["destination"] = getDestination()->toJSON();
  }

  result["callee"] = getCallee();
  llvm::SmallVector<llvm::json::Value> argsJson;

  for (const auto &arg : arguments) {
    argsJson.push_back(arg->toJSON());
  }

  result["args"] = llvm::json::Array(argsJson);

  addJSONProperties(result);
  return result;
}

bool ExternalFunctionCall::hasDestination() const {
  return destination != nullptr;
}

Expression *ExternalFunctionCall::getDestination() {
  assert(destination != nullptr && "Destination not set");
  return destination->cast<Expression>();
}

const Expression *ExternalFunctionCall::getDestination() const {
  assert(destination != nullptr && "Destination not set");
  return destination->cast<Expression>();
}

void ExternalFunctionCall::setDestination(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  destination = std::move(node);
  destination->setParent(this);
}

llvm::StringRef ExternalFunctionCall::getCallee() const {
  assert(callee != "" && "Callee not set");
  return callee;
}

void ExternalFunctionCall::setCallee(std::string newCallee) {
  callee = std::move(newCallee);
}

size_t ExternalFunctionCall::getNumOfArguments() const {
  return arguments.size();
}

Expression *ExternalFunctionCall::getArgument(size_t index) {
  assert(index < arguments.size());
  return arguments[index]->cast<Expression>();
}

const Expression *ExternalFunctionCall::getArgument(size_t index) const {
  assert(index < arguments.size());
  return arguments[index]->cast<Expression>();
}

llvm::ArrayRef<std::unique_ptr<ASTNode>>
ExternalFunctionCall::getArguments() const {
  return arguments;
}

void ExternalFunctionCall::setArguments(
    llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  arguments.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = arguments.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast
