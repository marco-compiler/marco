#include "marco/AST/BaseModelica/Subscript.h"
#include "marco/AST/BaseModelica/Expression.h"

using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
Subscript::Subscript(SourceRange location)
    : ASTNode(ASTNodeKind::Expression_Subscript, std::move(location)) {}

Subscript::Subscript(const Subscript &other) : ASTNode(other) {
  if (!other.isUnbounded()) {
    setExpression(other.getExpression()->clone());
  }
}

Subscript::~Subscript() = default;

std::unique_ptr<ASTNode> Subscript::clone() const {
  return std::make_unique<Subscript>(*this);
}

llvm::json::Value Subscript::toJSON() const {
  llvm::json::Object result;
  result["unbounded"] = isUnbounded();

  if (!isUnbounded()) {
    result["expression"] = getExpression()->toJSON();
  }

  addNodeKindToJSON(*this, result);
  return result;
}

bool Subscript::isUnbounded() const { return expression == nullptr; }

Expression *Subscript::getExpression() {
  assert(!isUnbounded());
  return expression->cast<Expression>();
}

const Expression *Subscript::getExpression() const {
  assert(!isUnbounded());
  return expression->cast<Expression>();
}

void Subscript::setExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  expression = std::move(node);
  expression->setParent(this);
}
} // namespace marco::ast::bmodelica
