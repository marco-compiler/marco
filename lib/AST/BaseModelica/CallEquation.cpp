#include "marco/AST/BaseModelica/CallEquation.h"
#include "marco/AST/BaseModelica/Expression.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
CallEquation::CallEquation(SourceRange location)
    : Equation(ASTNodeKind::Equation_Call, std::move(location)) {}

CallEquation::CallEquation(const CallEquation &other) : Equation(other) {
  setCall(other.call->clone());
}

CallEquation::~CallEquation() = default;

std::unique_ptr<ASTNode> CallEquation::clone() const {
  return std::make_unique<CallEquation>(*this);
}

llvm::json::Value CallEquation::toJSON() const {
  llvm::json::Object result;
  result["call"] = getCall()->toJSON();

  addNodeKindToJSON(*this, result);
  return result;
}

Expression *CallEquation::getCall() {
  assert(call != nullptr && "Call not set");
  return call->cast<Expression>();
}

const Expression *CallEquation::getCall() const {
  assert(call != nullptr && "Call not set");
  return call->cast<Expression>();
}

void CallEquation::setCall(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  call = std::move(node);
  call->setParent(this);
}
} // namespace marco::ast::bmodelica
