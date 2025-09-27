#include "marco/AST/BaseModelica/Node/AssignmentStatement.h"
#include "marco/AST/BaseModelica/Node/Expression.h"
#include "marco/AST/BaseModelica/Node/Tuple.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
AssignmentStatement::AssignmentStatement(SourceRange location)
    : Statement(ASTNode::Kind::Statement_Assignment, std::move(location)) {}

AssignmentStatement::AssignmentStatement(const AssignmentStatement &other)
    : Statement(other) {
  setDestinations(other.destinations->clone());
  setExpression(other.expression->clone());
}

AssignmentStatement::~AssignmentStatement() = default;

std::unique_ptr<ASTNode> AssignmentStatement::clone() const {
  return std::make_unique<AssignmentStatement>(*this);
}

llvm::json::Value AssignmentStatement::toJSON() const {
  llvm::json::Object result;
  result["destinations"] = getDestinations()->toJSON();
  result["expression"] = getExpression()->toJSON();

  addJSONProperties(result);
  return result;
}

Tuple *AssignmentStatement::getDestinations() {
  assert(destinations != nullptr && "Destinations not set");
  return destinations->cast<Tuple>();
}

const Tuple *AssignmentStatement::getDestinations() const {
  assert(destinations != nullptr && "Destinations not set");
  return destinations->cast<Tuple>();
}

void AssignmentStatement::setDestinations(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  destinations = std::move(node);

  if (!destinations->isa<Tuple>()) {
    auto tuple = std::make_unique<Tuple>(destinations->getLocation());
    tuple->setExpressions(destinations);
    destinations = std::move(tuple);
  }

  destinations->setParent(this);
}

Expression *AssignmentStatement::getExpression() {
  assert(expression != nullptr && "Expression not set");
  return expression->cast<Expression>();
}

const Expression *AssignmentStatement::getExpression() const {
  assert(expression != nullptr && "Expression not set");
  return expression->cast<Expression>();
}

void AssignmentStatement::setExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  expression = std::move(node);
  expression->setParent(this);
}
} // namespace marco::ast::bmodelica
