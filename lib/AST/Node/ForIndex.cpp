#include "marco/AST/Node/ForIndex.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ForIndex::ForIndex(SourceRange location)
      : ASTNode(ASTNode::Kind::ForIndex, std::move(location))
  {
  }

  ForIndex::ForIndex(const ForIndex& other)
      : ASTNode(other),
        name(other.name)
  {
    if (other.hasExpression()) {
      setExpression(other.getExpression()->clone());
    }
  }

  ForIndex::~ForIndex() = default;

  std::unique_ptr<ASTNode> ForIndex::clone() const
  {
    return std::make_unique<ForIndex>(*this);
  }

  llvm::json::Value ForIndex::toJSON() const
  {
    llvm::json::Object result;
    result["name"] = name;

    if (hasExpression()) {
      result["expression"] = getExpression()->toJSON();
    }

    addJSONProperties(result);
    return result;
  }

  llvm::StringRef ForIndex::getName() const
  {
    assert(!name.empty() && "Name not set");
    return name;
  }

  void ForIndex::setName(llvm::StringRef newName)
  {
    assert(!newName.empty() && "Empty name");
    name = newName.str();
  }

  bool ForIndex::hasExpression() const
  {
    return expression != nullptr;
  }

  Expression* ForIndex::getExpression()
  {
    assert(expression && "Expression not set");
    return expression->cast<Expression>();
  }

  const Expression* ForIndex::getExpression() const
  {
    assert(expression && "Expression not set");
    return expression->cast<Expression>();
  }

  void ForIndex::setExpression(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    expression = std::move(node);
    expression->setParent(this);
  }
}
