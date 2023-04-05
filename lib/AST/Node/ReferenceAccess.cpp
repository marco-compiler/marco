#include "marco/AST/Node/ReferenceAccess.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ReferenceAccess::ReferenceAccess(SourceRange location)
      : Expression(
          ASTNode::Kind::Expression_ReferenceAccess, std::move(location)),
        dummy(false)
  {
  }

  ReferenceAccess::ReferenceAccess(const ReferenceAccess& other)
      : Expression(other),
        name(other.name),
        dummy(other.dummy)
  {
  }

  ReferenceAccess::~ReferenceAccess() = default;

  std::unique_ptr<ASTNode> ReferenceAccess::clone() const
  {
    return std::make_unique<ReferenceAccess>(*this);
  }

  llvm::json::Value ReferenceAccess::toJSON() const
  {
    llvm::json::Object result;

    if (!isDummy()) {
      result["name"] = getName();
    }

    result["dummy"] = isDummy();

    addJSONProperties(result);
    return result;
  }

  bool ReferenceAccess::isLValue() const
  {
    return true;
  }

  llvm::StringRef ReferenceAccess::getName() const
  {
    return name;
  }

  void ReferenceAccess::setName(llvm::StringRef newName)
  {
    this->name = newName.str();
  }

  bool ReferenceAccess::isDummy() const
  {
    return dummy;
  }

  void ReferenceAccess::setDummy(bool value)
  {
    dummy = value;
  }
}
