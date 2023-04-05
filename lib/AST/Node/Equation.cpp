#include "marco/AST/Node/Equation.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Equation::Equation(SourceRange location)
      : ASTNode(ASTNode::Kind::Equation, std::move(location))
  {
  }

  Equation::Equation(const Equation& other)
      : ASTNode(other)
  {
    setLhsExpression(other.lhs->clone());
    setRhsExpression(other.rhs->clone());
  }

  Equation::~Equation() = default;

  std::unique_ptr<ASTNode> Equation::clone() const
  {
    return std::make_unique<Equation>(*this);
  }

  llvm::json::Value Equation::toJSON() const
  {
    llvm::json::Object result;
    result["lhs"] = getLhsExpression()->toJSON();
    result["rhs"] = getRhsExpression()->toJSON();

    addJSONProperties(result);
    return result;
  }

  Expression* Equation::getLhsExpression()
  {
    assert(lhs != nullptr && "Left-hand side expression not set");
    return lhs->cast<Expression>();
  }

  const Expression* Equation::getLhsExpression() const
  {
    assert(lhs != nullptr && "Left-hand side expression not set");
    return lhs->cast<Expression>();
  }

  void Equation::setLhsExpression(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    lhs = std::move(node);
    lhs->setParent(this);
  }

  Expression* Equation::getRhsExpression()
  {
    assert(rhs != nullptr && "Right-hand side expression not set");
    return rhs->cast<Expression>();
  }

  const Expression* Equation::getRhsExpression() const
  {
    assert(rhs != nullptr && "Right-hand side expression not set");
    return rhs->cast<Expression>();
  }

  void Equation::setRhsExpression(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    rhs = std::move(node);
    rhs->setParent(this);
  }
}
