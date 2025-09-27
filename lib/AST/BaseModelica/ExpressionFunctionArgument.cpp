#include "marco/AST/BaseModelica/ExpressionFunctionArgument.h"
#include "marco/AST/BaseModelica/Expression.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
ExpressionFunctionArgument::ExpressionFunctionArgument(SourceRange location)
    : FunctionArgument(ASTNodeKind::FunctionArgument_Expression,
                       std::move(location)) {}

ExpressionFunctionArgument::ExpressionFunctionArgument(
    const ExpressionFunctionArgument &other)
    : FunctionArgument(other) {
  setExpression(other.expression->clone());
}

ExpressionFunctionArgument::~ExpressionFunctionArgument() = default;

std::unique_ptr<ASTNode> ExpressionFunctionArgument::clone() const {
  return std::make_unique<ExpressionFunctionArgument>(*this);
}

llvm::json::Value ExpressionFunctionArgument::toJSON() const {
  llvm::json::Object result;
  result["expression"] = getExpression()->toJSON();

  addNodeKindToJSON(*this, result);
  return result;
}

Expression *ExpressionFunctionArgument::getExpression() {
  assert(expression && "Expression not set");
  return expression->cast<Expression>();
}

const Expression *ExpressionFunctionArgument::getExpression() const {
  assert(expression && "Expression not set");
  return expression->cast<Expression>();
}

void ExpressionFunctionArgument::setExpression(std::unique_ptr<ASTNode> node) {
  assert(node->isa<Expression>());
  expression = std::move(node);
  expression->setParent(this);
}
} // namespace marco::ast::bmodelica
