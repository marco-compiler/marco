#include "marco/AST/Node/Member.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Modification.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Member::Member(
      SourceRange location,
      llvm::StringRef name,
      Type type,
      TypePrefix typePrefix,
      bool isPublic,
      std::unique_ptr<Modification> modification)
      : ASTNode(std::move(location)),
        name(name.str()),
        type(std::move(type)),
        typePrefix(std::move(typePrefix)),
        isPublicMember(isPublic),
        modification(std::move(modification))
  {
  }

  Member::Member(const Member& other)
      : ASTNode(other),
        name(other.name),
        type(other.type),
        typePrefix(other.typePrefix),
        isPublicMember(other.isPublicMember)
  {
    if (other.hasModification()) {
      modification = std::make_unique<Modification>(*other.modification);
    }
  }

  Member::Member(Member&& other) = default;

  Member::~Member() = default;

  Member& Member::operator=(const Member& other)
  {
    Member result(other);
    swap(*this, result);
    return *this;
  }

  Member& Member::operator=(Member&& other) = default;

  void swap(Member& first, Member& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    swap(first.name, second.name);
    swap(first.type, second.type);
    swap(first.typePrefix, second.typePrefix);
    swap(first.isPublicMember, second.isPublicMember);
    swap(first.modification, second.modification);
  }

  void Member::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "member: {name: " << name << ", type: ";
    type.print(os);
    os << "}\n";

    if (hasModification()) {
      os.indent(indents + 1);
      os << "modification:\n";
      modification->print(os, indents + 2);
    }
  }

  bool Member::operator==(const Member& other) const
  {
    return name == other.name &&
        type == other.type &&
        modification == other.modification;
  }

  bool Member::operator!=(const Member& other) const
  {
    return !(*this == other);
  }

  llvm::StringRef Member::getName() const
  {
    return name;
  }

  Type& Member::getType()
  {
    return type;
  }

  const Type& Member::getType() const
  {
    return type;
  }

  void Member::setType(Type new_type)
  {
    type = std::move(new_type);
  }

  TypePrefix Member::getTypePrefix() const
  {
    return typePrefix;
  }

  bool Member::isPublic() const
  {
    return isPublicMember;
  }

  bool Member::isParameter() const
  {
    return typePrefix.isParameter();
  }

  bool Member::isInput() const
  {
    return typePrefix.isInput();
  }

  bool Member::isOutput() const
  {
    return typePrefix.isOutput();
  }

  bool Member::hasModification() const
  {
    return modification != nullptr;
  }

  Modification* Member::getModification()
  {
    assert(hasModification());
    return modification.get();
  }

  const Modification* Member::getModification() const
  {
    assert(hasModification());
    return modification.get();
  }

  bool Member::hasExpression() const
  {
    if (!hasModification()) {
      return false;
    }

    if (!modification->hasExpression()) {
      return false;
    }

    return modification->getExpression();
  }

  Expression* Member::getExpression()
  {
    assert(hasExpression());
    return modification->getExpression();
  }

  const Expression* Member::getExpression() const
  {
    assert(hasExpression());
    return modification->getExpression();
  }

  bool Member::hasStartProperty() const
  {
    if (!hasModification()) {
      return false;
    }

    return modification->hasStartProperty();
  }

  StartModificationProperty Member::getStartProperty() const
  {
    assert(hasStartProperty());
    return modification->getStartProperty();
  }

  bool Member::getFixedProperty() const
  {
    if (!hasModification()) {
      return false;
    }

    return modification->getFixedProperty();
  }

  StartModificationProperty::StartModificationProperty(bool each, const Expression& value)
    : each(each), value(&value)
  {
  }
}
