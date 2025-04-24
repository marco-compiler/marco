#include "marco/AST/Node/CallStatement.h"
#include "marco/AST/Node/Call.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast {
CallStatement::CallStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_Call, std::move(location)) {}

CallStatement::CallStatement(const CallStatement &other) : Statement(other) {
  setCall(other.call->clone());
}

std::unique_ptr<ASTNode> CallStatement::clone() const {
  return std::make_unique<CallStatement>(*this);
}

llvm::json::Value CallStatement::toJSON() const {
  llvm::json::Object result;

  result["call"] = getCall()->toJSON();

  addJSONProperties(result);
  return result;
}

Call *CallStatement::getCall() {
  assert(call != nullptr && "Call not set");
  return call->cast<Call>();
}

const Call *CallStatement::getCall() const {
  assert(call != nullptr && "Call not set");
  return call->cast<Call>();
}

void CallStatement::setCall(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Call>());
  call = std::move(node);
  call->setParent(this);
}
} // namespace marco::ast
