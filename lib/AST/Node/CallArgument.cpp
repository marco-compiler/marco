#include "marco/AST/Node/CallArgument.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  CallArgument::CallArgument(SourceRange location)
      : ASTNode(ASTNode::Kind::CallArgument, std::move(location))
  {
  }

  CallArgument::CallArgument(const CallArgument& other)
      : ASTNode(other),
        name(other.name),
        value(other.value->clone())
  {
  }

  CallArgument::~CallArgument() = default;

  std::unique_ptr<ASTNode> CallArgument::clone() const
  {
    return std::make_unique<CallArgument>(*this);
  }

  llvm::json::Value CallArgument::toJSON() const
  {
    llvm::json::Object result;

    if (isNamed()) {
      result["name"] = name;
    }

    result["value"] = getValue()->toJSON();

    addJSONProperties(result);
    return result;
  }

  bool CallArgument::isNamed() const
  {
    return name.has_value();
  }

  llvm::StringRef CallArgument::getName() const
  {
    assert(name && "Name not set");
    return *name;
  }

  void CallArgument::setName(llvm::StringRef newName)
  {
    name = newName.str();
  }

  Expression* CallArgument::getValue()
  {
    assert(value && "Value not set");
    return value->cast<Expression>();
  }

  const Expression* CallArgument::getValue() const
  {
    assert(value && "Value not set");
    return value->cast<Expression>();
  }

  void CallArgument::setValue(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    value = std::move(node);
    value->setParent(this);
  }
}
