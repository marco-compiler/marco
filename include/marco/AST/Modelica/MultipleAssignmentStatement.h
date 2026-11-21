#ifndef MARCO_AST_MODELICA_MULTIPLEASSIGNMENTSTATEMENT_H
#define MARCO_AST_MODELICA_MULTIPLEASSIGNMENTSTATEMENT_H

#include "marco/AST/Modelica/Call.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// Assignment of multiple outputs `(a, , b) := f(x)`.
class MultipleAssignmentStatement
    : public NodeImpl<MultipleAssignmentStatement, Statement,
                      NodeKind::Statement_MultipleAssignment> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  /// Get the destinations. Omitted destinations are null.
  ChildrenRef<Expression> getDestinations() const;

  void addDestination(std::unique_ptr<Expression> node);

  Call *getCall() const;

  void setCall(std::unique_ptr<Call> node);

private:
  Children<Expression> destinations;
  Child<Call> call;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_MULTIPLEASSIGNMENTSTATEMENT_H
