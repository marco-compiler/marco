#include "marco/AST/Node/Modification.h"
#include "marco/AST/Node/Constant.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Member.h"
#include "marco/AST/Node/ArrayConstant.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Modification::Modification(
      SourceRange location)
    : ASTNode(ASTNode::Kind::Modification, std::move(location))
  {
  }

  Modification::Modification(const Modification& other)
    : ASTNode(other)
  {
    if (other.hasClassModification()) {
      setClassModification(other.classModification->clone());
    }

    if (other.hasExpression()) {
      setExpression(other.expression->clone());
    }
  }

  Modification::~Modification() = default;

  std::unique_ptr<ASTNode> Modification::clone() const
  {
    return std::make_unique<Modification>(*this);
  }

  llvm::json::Value Modification::toJSON() const
  {
    llvm::json::Object result;

    if (hasClassModification()) {
      result["class_modification"] = getClassModification()->toJSON();
    }

    if (hasExpression()) {
      result["expression"] = getExpression()->toJSON();
    }

    addJSONProperties(result);
    return result;
  }

  bool Modification::hasClassModification() const
  {
    return classModification != nullptr;
  }

  ClassModification* Modification::getClassModification()
  {
    assert(hasClassModification() && "Class modification not set");
    return classModification->cast<ClassModification>();
  }

  const ClassModification* Modification::getClassModification() const
  {
    assert(hasClassModification() && "Class modification not set");
    return classModification->cast<ClassModification>();
  }

  void Modification::setClassModification(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<ClassModification>());
    classModification = std::move(node);
    classModification->setParent(this);
  }

  bool Modification::hasExpression() const
  {
    return expression != nullptr;
  }

  Expression* Modification::getExpression()
  {
    assert(hasExpression() && "Expression not set");
    return expression->cast<Expression>();
  }

  const Expression* Modification::getExpression() const
  {
    assert(hasExpression() && "Expression not set");
    return expression->cast<Expression>();
  }

  void Modification::setExpression(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    expression = std::move(node);
    expression->setParent(this);
  }

  bool Modification::hasStartExpression() const
  {
    if (!hasClassModification()) {
      return false;
    }

    return getClassModification()->hasStartExpression();
  }

  Expression* Modification::getStartExpression()
  {
    assert(hasStartExpression());
    return getClassModification()->getStartExpression();
  }

  const Expression* Modification::getStartExpression() const
  {
    assert(hasStartExpression());
    return getClassModification()->getStartExpression();
  }

  bool Modification::getFixedProperty() const
  {
    if (!hasClassModification()) {
      return false;
    }

    return getClassModification()->getFixedProperty();
  }

  bool Modification::getEachProperty() const
  {
    if (!hasClassModification()) {
      return false;
    }

    return getClassModification()->getEachProperty();
  }

  ClassModification::ClassModification(SourceRange location)
    : ASTNode(ASTNode::Kind::ClassModification, std::move(location))
  {
    for (const auto& arg : arguments) {
      this->arguments.push_back(arg->clone());
    }
  }

  ClassModification::ClassModification(const ClassModification& other)
    : ASTNode(other)
  {
    for (const auto& arg : other.arguments) {
      this->arguments.push_back(arg->clone());
    }
  }

  ClassModification::~ClassModification() = default;

  std::unique_ptr<ASTNode> ClassModification::clone() const
  {
    return std::make_unique<ClassModification>(*this);
  }

  llvm::json::Value ClassModification::toJSON() const
  {
    llvm::json::Object result;

    llvm::SmallVector<llvm::json::Value> argumentsJson;

    for (const auto& argument : arguments) {
      argumentsJson.push_back(argument->toJSON());
    }

    result["arguments"] = llvm::json::Array(argumentsJson);

    addJSONProperties(result);
    return result;
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>>
  ClassModification::getArguments() const
  {
    return arguments;
  }

  void ClassModification::setArguments(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    arguments.clear();

    for (const auto& node : nodes) {
      assert(node->isa<Argument>());
      auto& clone = arguments.emplace_back(node->clone());
      clone->setParent(this);
    }
  }

  bool ClassModification::hasStartExpression() const
  {
    for (const auto& argument : arguments) {
      if (!argument->isa<ElementModification>()) {
        continue;
      }

      auto* elementModification = argument->cast<ElementModification>();

      if (elementModification->getName() != "start") {
        continue;
      }

      assert(elementModification->hasModification());
      return true;
    }

    return false;
  }

  Expression* ClassModification::getStartExpression()
  {
    assert(hasStartExpression());

    for (auto& argument : arguments) {
      if (!argument->isa<ElementModification>()) {
        continue;
      }

      auto* elementModification = argument->cast<ElementModification>();

      if (elementModification->getName() != "start") {
        continue;
      }

      return elementModification->getModification()->getExpression();
    }

    llvm_unreachable("Start property not found");
  }

  const Expression* ClassModification::getStartExpression() const
  {
    assert(hasStartExpression());

    for (const auto& argument : arguments) {
      if (!argument->isa<ElementModification>()) {
        continue;
      }

      auto* elementModification = argument->cast<ElementModification>();

      if (elementModification->getName() != "start") {
        continue;
      }

      return elementModification->getModification()->getExpression();
    }

    llvm_unreachable("Start property not found");
  }

  std::optional<bool> ClassModification::isArrayUniformConstBool(const ArrayConstant *array)
  {
    size_t elements = array->size();
    std::optional<bool> lastValue = std::nullopt;

    for (size_t i = 0; i < elements; i++) {
      const Expression *exp = (*array)[i];
      bool value;
      if (exp->isa<ArrayConstant>()) {
        auto tmp = isArrayUniformConstBool(exp->cast<ArrayConstant>());
        if (!tmp) {
          return std::nullopt;
        } else {
          value = tmp.value();
        }
      } else if (exp->isa<Constant>()) {
        value = exp->cast<Constant>()->as<bool>();
      } else {
        return std::nullopt;
      }
      if (!lastValue)
        lastValue = value;
      else if (lastValue.value() != value)
        return std::nullopt;
    }

    return lastValue;
  }

  bool ClassModification::getFixedProperty() const
  {
    for (const auto& argument : arguments) {
      if (!argument->isa<ElementModification>()) {
        continue;
      }

      const auto* elementModification = argument->cast<ElementModification>();

      if (elementModification->getName() != "fixed") {
        continue;
      }

      assert(elementModification->hasModification());
      const auto* modification = elementModification->getModification();
      assert(modification->hasExpression());
      const auto* modificationExpression = modification->getExpression();
      if (modificationExpression->isa<Constant>()) {
        return modificationExpression->cast<Constant>()->as<bool>();
      } else if (modificationExpression->isa<ArrayConstant>()) {
        // FIXME: Handle this case without special casing
        // fixed = {{{{true, true}, {true, true}}, {{true, true}, {true, ...
        const auto *array = modificationExpression->cast<ArrayConstant>();
        auto val = isArrayUniformConstBool(array);
        assert(val);
        return val.value();
      }
      assert(false);
      return false;
    }

    return false;
  }

  bool ClassModification::getEachProperty() const
  {
    for (const auto& argument : arguments) {
      if (!argument->isa<ElementModification>()) {
        continue;
      }

      auto* elementModification = argument->cast<ElementModification>();
      return elementModification->hasEachProperty();
    }

    return false;
  }

  ElementModification::ElementModification(SourceRange location)
    : Argument(ASTNode::Kind::Argument_ElementModification, std::move(location)),
      each(false),
      final(false),
      name("")
  {
  }

  ElementModification::ElementModification(const ElementModification& other)
      : Argument(other),
        each(other.each),
        final(other.final),
        name(other.name)
  {
    if (other.hasModification()) {
      setModification(other.modification->clone());
    }
  }

  ElementModification::~ElementModification() = default;

  std::unique_ptr<ASTNode> ElementModification::clone() const
  {
    return std::make_unique<ElementModification>(*this);
  }

  llvm::json::Value ElementModification::toJSON() const
  {
    llvm::json::Object result;
    result["each"] = hasEachProperty();
    result["final"] = hasFinalProperty();
    result["name"] = getName();

    if (hasModification()) {
      result["modification"] = getModification()->toJSON();
    }

    addJSONProperties(result);
    return result;
  }

  bool ElementModification::hasEachProperty() const
  {
    return each;
  }

  void ElementModification::setEachProperty(bool value)
  {
    each = value;
  }

  bool ElementModification::hasFinalProperty() const
  {
    return final;
  }

  void ElementModification::setFinalProperty(bool value)
  {
    final = value;
  }

  llvm::StringRef ElementModification::getName() const
  {
    return name;
  }

  void ElementModification::setName(llvm::StringRef newName)
  {
    name = newName.str();
  }

  bool ElementModification::hasModification() const
  {
    return modification != nullptr;
  }

  Modification* ElementModification::getModification()
  {
    assert(hasModification() && "Modification not set");
    return modification->cast<Modification>();
  }

  const Modification* ElementModification::getModification() const
  {
    assert(hasModification() && "Modification not set");
    return modification->cast<Modification>();
  }

  void ElementModification::setModification(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Modification>());
    modification = std::move(node);
    modification->setParent(this);
  }

  ElementReplaceable::ElementReplaceable(SourceRange location)
      : Argument(ASTNode::Kind::Argument_ElementReplaceable, std::move(location))
  {
  }

  ElementReplaceable::ElementReplaceable(const ElementReplaceable& other)
      : Argument(other)
  {
  }

  ElementReplaceable::~ElementReplaceable() = default;

  std::unique_ptr<ASTNode> ElementReplaceable::clone() const
  {
    return std::make_unique<ElementReplaceable>(*this);
  }

  llvm::json::Value ElementReplaceable::toJSON() const
  {
    llvm::json::Object result;
    addJSONProperties(result);
    return result;
  }

  ElementRedeclaration::ElementRedeclaration(SourceRange location)
      : Argument(ASTNode::Kind::Argument_ElementRedeclaration, std::move(location))
  {
  }

  ElementRedeclaration::ElementRedeclaration(const ElementRedeclaration& other)
      : Argument(other)
  {
  }

  ElementRedeclaration::~ElementRedeclaration() = default;

  std::unique_ptr<ASTNode> ElementRedeclaration::clone() const
  {
    return std::make_unique<ElementRedeclaration>(*this);
  }

  llvm::json::Value ElementRedeclaration::toJSON() const
  {
    llvm::json::Object result;
    addJSONProperties(result);
    return result;
  }
}
