#ifndef MARCO_AST_MODELICA_EXTERNALCLAUSE_H
#define MARCO_AST_MODELICA_EXTERNALCLAUSE_H

#include "marco/AST/Modelica/ClassModification.h"
#include "marco/AST/Modelica/ComponentReference.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// External function clause.
class ExternalClause
    : public NodeImpl<ExternalClause, Node, NodeKind::ExternalClause> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Get the language specification, if any.
  const std::optional<std::string> &getLanguage() const;

  void setLanguage(std::optional<std::string> value);

  /// Whether the external function call is specified.
  bool hasFunctionCall() const;

  llvm::StringRef getFunction() const;

  void setFunction(std::string name);

  /// Get the component reference receiving the result, if any.
  ComponentReference *getOutput() const;

  void setOutput(std::unique_ptr<ComponentReference> node);

  ChildrenRef<Expression> getArguments() const;

  void addArgument(std::unique_ptr<Expression> node);

  ClassModification *getAnnotation() const;

  void setAnnotation(std::unique_ptr<ClassModification> node);

private:
  std::optional<std::string> language;
  std::string function;
  Child<ComponentReference> output;
  Children<Expression> arguments;
  Child<ClassModification> annotation;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EXTERNALCLAUSE_H
