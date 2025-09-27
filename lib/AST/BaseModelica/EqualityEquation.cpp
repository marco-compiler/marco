#include "marco/AST/BaseModelica/EqualityEquation.h"
#include "marco/AST/BaseModelica/Expression.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
EqualityEquation::EqualityEquation(SourceRange location)
    : ASTNode(ASTNodeKind::Equation_Equality, std::move(location)) {}

EqualityEquation::EqualityEquation(const EqualityEquation &other)
    : ASTNode(other) {
  setLhsExpression(other.lhs->clone());
  setRhsExpression(other.rhs->clone());
}

EqualityEquation::~EqualityEquation() = default;

std::unique_ptr<ASTNode> EqualityEquation::clone() const {
  return std::make_unique<EqualityEquation>(*this);
}

llvm::json::Value EqualityEquation::toJSON() const {
  llvm::json::Object result;
  result["lhs"] = getLhsExpression()->toJSON();
  result["rhs"] = getRhsExpression()->toJSON();

  addNodeKindToJSON(*this, result);
  return result;
}

Expression *EqualityEquation::getLhsExpression() {
  assert(lhs != nullptr && "Left-hand side expression not set");
  return lhs->cast<Expression>();
}

const Expression *EqualityEquation::getLhsExpression() const {
  assert(lhs != nullptr && "Left-hand side expression not set");
  return lhs->cast<Expression>();
}

void EqualityEquation::setLhsExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  lhs = std::move(node);
  lhs->setParent(this);
}

Expression *EqualityEquation::getRhsExpression() {
  assert(rhs != nullptr && "Right-hand side expression not set");
  return rhs->cast<Expression>();
}

const Expression *EqualityEquation::getRhsExpression() const {
  assert(rhs != nullptr && "Right-hand side expression not set");
  return rhs->cast<Expression>();
}

void EqualityEquation::setRhsExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  rhs = std::move(node);
  rhs->setParent(this);
}
} // namespace marco::ast::bmodelica
