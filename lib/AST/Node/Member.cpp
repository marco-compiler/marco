#include "marco/AST/Node/Member.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Modification.h"
#include "marco/AST/Node/Type.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Member::Member(SourceRange location)
      : ASTNode(ASTNode::Kind::Member, std::move(location)),
        name(""),
        isPublicMember(true)
  {
  }

  Member::Member(const Member& other)
      : ASTNode(other),
        name(other.name),
        isPublicMember(other.isPublicMember)
  {
    setType(other.type->clone());
    setTypePrefix(other.typePrefix->clone());

    if (other.hasModification()) {
      setModification(other.modification->clone());
    }
  }

  Member::~Member() = default;

  std::unique_ptr<ASTNode> Member::clone() const
  {
    return std::make_unique<Member>(*this);
  }

  llvm::json::Value Member::toJSON() const
  {
    llvm::json::Object result;
    result["name"] = getName();
    result["type"] = getType()->toJSON();
    result["type_prefix"] = getTypePrefix()->toJSON();
    result["public"] = isPublicMember;

    if (hasModification()) {
      result["modification"] = getModification()->toJSON();
    }

    addJSONProperties(result);
    return result;
  }

  llvm::StringRef Member::getName() const
  {
    return name;
  }

  void Member::setName(llvm::StringRef newName)
  {
    name = newName.str();
  }

  VariableType* Member::getType()
  {
    assert(type != nullptr && "Type not set");
    return type->cast<VariableType>();
  }

  const VariableType* Member::getType() const
  {
    assert(type != nullptr && "Type not set");
    return type->cast<VariableType>();
  }

  void Member::setType(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<VariableType>());
    type = std::move(node);
    type->setParent(this);
  }

  TypePrefix* Member::getTypePrefix()
  {
    return typePrefix->cast<TypePrefix>();
  }

  const TypePrefix* Member::getTypePrefix() const
  {
    return typePrefix->cast<TypePrefix>();
  }

  void Member::setTypePrefix(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<TypePrefix>());
    typePrefix = std::move(node);
    typePrefix->setParent(this);
  }

  bool Member::isPublic() const
  {
    return isPublicMember;
  }

  void Member::setPublic(bool value)
  {
    isPublicMember = value;
  }

  bool Member::isDiscrete() const
  {
    return getTypePrefix()->isDiscrete();
  }

  bool Member::isParameter() const
  {
    return getTypePrefix()->isParameter();
  }

  bool Member::isConstant() const
  {
    return getTypePrefix()->isConstant();
  }

  bool Member::isInput() const
  {
    return getTypePrefix()->isInput();
  }

  bool Member::isOutput() const
  {
    return getTypePrefix()->isOutput();
  }

  bool Member::hasModification() const
  {
    return modification != nullptr;
  }

  Modification* Member::getModification()
  {
    assert(hasModification());
    return modification->cast<Modification>();
  }

  const Modification* Member::getModification() const
  {
    assert(hasModification());
    return modification->cast<Modification>();
  }

  void Member::setModification(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Modification>());
    modification = std::move(node);
    modification->setParent(this);
  }

  bool Member::hasExpression() const
  {
    if (!hasModification()) {
      return false;
    }

    if (!getModification()->hasExpression()) {
      return false;
    }

    return getModification()->getExpression();
  }

  Expression* Member::getExpression()
  {
    assert(hasExpression());
    return getModification()->getExpression();
  }

  const Expression* Member::getExpression() const
  {
    assert(hasExpression());
    return getModification()->getExpression();
  }

  bool Member::hasStartExpression() const
  {
    if (!hasModification()) {
      return false;
    }

    return getModification()->hasStartExpression();
  }

  Expression* Member::getStartExpression()
  {
    assert(hasStartExpression());
    return getModification()->getStartExpression();
  }

  const Expression* Member::getStartExpression() const
  {
    assert(hasStartExpression());
    return getModification()->getStartExpression();
  }

  bool Member::getFixedProperty() const
  {
    if (!hasModification()) {
      return false;
    }

    return getModification()->getFixedProperty();
  }

  bool Member::getEachProperty() const
  {
    if (!hasModification()) {
      return false;
    }

    return getModification()->getEachProperty();
  }
}
