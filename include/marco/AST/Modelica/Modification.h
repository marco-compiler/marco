#ifndef MARCO_AST_MODELICA_MODIFICATION_H
#define MARCO_AST_MODELICA_MODIFICATION_H

#include "marco/AST/Modelica/ClassModification.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Modification: an optional class modification, followed by
/// an optional value (`= expression` or `= break`).
class Modification
    : public NodeImpl<Modification, Node, NodeKind::Modification> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ClassModification *getClassModification() const;

  void setClassModification(std::unique_ptr<ClassModification> node);

  /// Get the value, if any.
  Expression *getValue() const;

  void setValue(std::unique_ptr<Expression> node);

  /// Whether the value is `break`.
  bool isBreak() const;

  void setBreak(bool isBreak);

private:
  Child<ClassModification> classModification;
  Child<Expression> value;
  bool breakValue = false;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_MODIFICATION_H
