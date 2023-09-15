#include "marco/AST/Node/ExpressionFunctionArgument.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ExpressionFunctionArgument::ExpressionFunctionArgument(SourceRange location)
      : FunctionArgument(ASTNode::Kind::FunctionArgument_Expression,
                         std::move(location))
  {
  }

  ExpressionFunctionArgument::ExpressionFunctionArgument(
      const ExpressionFunctionArgument& other)
      : FunctionArgument(other)
  {
    setExpression(other.expression->clone());
  }

  ExpressionFunctionArgument::~ExpressionFunctionArgument() = default;

  std::unique_ptr<ASTNode> ExpressionFunctionArgument::clone() const
  {
    return std::make_unique<ExpressionFunctionArgument>(*this);
  }

  llvm::json::Value ExpressionFunctionArgument::toJSON() const
  {
    llvm::json::Object result;
    result["expression"] = getExpression()->toJSON();

    addJSONProperties(result);
    return result;
  }

  Expression* ExpressionFunctionArgument::getExpression()
  {
    assert(expression && "Expression not set");
    return expression->cast<Expression>();
  }

  const Expression* ExpressionFunctionArgument::getExpression() const
  {
    assert(expression && "Expression not set");
    return expression->cast<Expression>();
  }

  void ExpressionFunctionArgument::setExpression(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    expression = std::move(node);
    expression->setParent(this);
  }
}
