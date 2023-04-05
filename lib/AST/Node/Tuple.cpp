#include "marco/AST/Node/Tuple.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Tuple::Tuple(SourceRange location)
      : Expression(ASTNode::Kind::Expression_Tuple, std::move(location))
  {
  }

  Tuple::Tuple(const Tuple& other)
      : Expression(other)
  {
    setExpressions(other.expressions);
  }

  Tuple::~Tuple() = default;

  std::unique_ptr<ASTNode> Tuple::clone() const
  {
    return std::make_unique<Tuple>(*this);
  }

  llvm::json::Value Tuple::toJSON() const
  {
    llvm::json::Object result;

    llvm::SmallVector<llvm::json::Value> expressionsJson;

    for (const auto& expression : expressions) {
      expressionsJson.push_back(expression->toJSON());
    }

    result["expressions"] = llvm::json::Array(expressionsJson);

    addJSONProperties(result);
    return result;
  }

  bool Tuple::isLValue() const
  {
    return false;
  }

  size_t Tuple::size() const
  {
    return expressions.size();
  }

  Expression* Tuple::getExpression(size_t index)
  {
    assert(index < expressions.size());
    return expressions[index]->cast<Expression>();
  }

  const Expression* Tuple::getExpression(size_t index) const
  {
    assert(index < expressions.size());
    return expressions[index]->cast<Expression>();
  }

  void Tuple::setExpressions(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    expressions.clear();

    for (const auto& node : nodes) {
      assert(node->isa<Expression>());
      auto& clone = expressions.emplace_back(node->clone());
      clone->setParent(this);
    }
  }
}
