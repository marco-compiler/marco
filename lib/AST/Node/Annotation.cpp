#include "marco/AST/Node/Annotation.h"
#include "marco/AST/Node/Call.h"
#include "marco/AST/Node/Constant.h"
#include "marco/AST/Node/Expression.h"
#include "marco/AST/Node/Modification.h"
#include "marco/AST/Node/Function.h"
#include "marco/AST/Node/ReferenceAccess.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Annotation::Annotation(SourceRange location)
      : ASTNode(ASTNode::Kind::Annotation, std::move(location))
  {
  }

  Annotation::Annotation(const Annotation& other)
      : ASTNode(other)
  {
    setProperties(other.properties->clone());
  }

  Annotation::~Annotation() = default;

  std::unique_ptr<ASTNode> Annotation::clone() const
  {
    return std::make_unique<Annotation>(*this);
  }

  llvm::json::Value Annotation::toJSON() const
  {
    llvm::json::Object result;
    result["properties"] = getProperties()->toJSON();

    addJSONProperties(result);
    return result;
  }

  ClassModification* Annotation::getProperties()
  {
    assert(properties != nullptr && "Properties not set");
    return properties->cast<ClassModification>();
  }

  const ClassModification* Annotation::getProperties() const
  {
    assert(properties != nullptr && "Properties not set");
    return properties->cast<ClassModification>();
  }

  void Annotation::setProperties(std::unique_ptr<ASTNode> newProperties)
  {
    assert(newProperties->isa<ClassModification>());
    properties = std::move(newProperties);
    properties->setParent(this);
  }

  /// Inline property AST structure:
  ///
  ///          class-modification
  ///                  |
  ///            argument-list
  ///             /         \
  ///       argument         ...
  ///          |
  ///  element-modification
  ///    /           \
  ///  name        modification
  /// inline           |
  ///              expression
  ///                true
  bool Annotation::getInlineProperty() const
  {
    for (const auto& argument : getProperties()->getArguments()) {
      if (auto* elementModification = argument->dyn_cast<ElementModification>()) {
        if (elementModification->getName().equals_insensitive("inline") && elementModification->hasModification()) {
          const auto& modification = elementModification->getModification();
          return modification->getExpression()->cast<Constant>()->as<bool>();
        }
      }
    }

    return false;
  }

  bool Annotation::hasDerivativeAnnotation() const
  {
    for (const auto& argument : getProperties()->getArguments()) {
      if (auto* elementModification = argument->dyn_cast<ElementModification>()) {
        if (elementModification->getName() == "derivative") {
          return true;
        }
      }
    }

    return false;
  }

  /// Derivative property AST structure:
  ///
  ///          class-modification
  ///                  |
  ///            argument-list
  ///             /         \
  ///       argument         ...
  ///          |
  ///  element-modification
  ///    /                \
  ///  name            modification
  /// derivative       /         \
  ///             expression   class-modification
  ///               foo                |
  ///                           argument-list
  ///                             /       \
  ///                        argument     ...
  ///                           |
  ///                   element-modification
  ///                    /              \
  ///                  name          modification
  ///                  order             |
  ///                                expression
  ///                                  <int>
  DerivativeAnnotation Annotation::getDerivativeAnnotation() const
  {
    assert(hasDerivativeAnnotation());

    for (const auto& argument : getProperties()->getArguments()) {
      if (auto* elementModification = argument->dyn_cast<ElementModification>()) {
        if (elementModification->getName() != "derivative") {
          continue;
        }

        auto* modification = elementModification->getModification();
        auto name = modification->getExpression()->cast<ReferenceAccess>()->getName();
        unsigned int order = 1;

        if (modification->hasClassModification()) {
          for (const auto& derivativeArgument : modification->getClassModification()->getArguments()) {
            if (auto* derivativeModification = derivativeArgument->dyn_cast<ElementModification>()) {
              if (derivativeModification->getName() == "order") {
                order = derivativeModification->getModification()->getExpression()->dyn_cast<Constant>()->as<int64_t>();
              }
            }
          }
        }

        return DerivativeAnnotation(name, order);
      }
    }

    // Normally unreachable
    return DerivativeAnnotation("", 1);
  }

  InverseFunctionAnnotation Annotation::getInverseFunctionAnnotation() const
  {
    InverseFunctionAnnotation result;

    for (const auto& argument : getProperties()->getArguments()) {
      if (auto* elementModification = argument->dyn_cast<ElementModification>()) {
        if (elementModification->getName().equals_insensitive("inverse")) {
          assert(elementModification->hasModification());
          const auto& modification = elementModification->getModification();
          assert(modification->hasClassModification());

          for (const auto& inverseDeclaration : modification->getClassModification()->getArguments()) {
            const auto& inverseDeclarationFullExpression = inverseDeclaration->cast<ElementModification>();
            assert(inverseDeclarationFullExpression->hasModification());
            const auto& callExpression = inverseDeclarationFullExpression->getModification();
            assert(callExpression->hasExpression());
            const auto* call = callExpression->getExpression()->cast<Call>();

            llvm::SmallVector<std::string, 3> args;

            for (const auto& arg : call->getArguments()) {
              args.push_back(arg->cast<ReferenceAccess>()->getName());
            }

            result.addInverse(
                inverseDeclarationFullExpression->getName().str(),
                call->getCallee()->cast<ReferenceAccess>()->getName(),
                args);
          }
        }
      }
    }

    return result;
  }
}
