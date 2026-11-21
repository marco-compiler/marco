#ifndef MARCO_AST_MODELICA_ASSIGNMENTSTATEMENT_H
#define MARCO_AST_MODELICA_ASSIGNMENTSTATEMENT_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// Assignment `destination := value`. The destination is a component
/// reference or a `der` call on a component reference.
class AssignmentStatement
    : public NodeImpl<AssignmentStatement, Statement,
                      NodeKind::Statement_Assignment> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Expression *getDestination() const;

  void setDestination(std::unique_ptr<Expression> node);

  Expression *getValue() const;

  void setValue(std::unique_ptr<Expression> node);

private:
  Child<Expression> destination;
  Child<Expression> value;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ASSIGNMENTSTATEMENT_H
