#ifndef MARCO_AST_MODELICA_IFEQUATION_H
#define MARCO_AST_MODELICA_IFEQUATION_H

#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// If-equation. Each branch has a condition, except for
/// the optional else branch.
class IfEquation : public NodeImpl<IfEquation, Equation, NodeKind::Equation_If> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ChildrenRef<Expression> getConditions() const;

  size_t getNumBranches() const;

  /// Get the equations of a branch.
  ChildrenRef<Equation> getBranch(size_t index) const;

  void addBranch(std::unique_ptr<Expression> condition,
                 Children<Equation> equations);

  ChildrenRef<Equation> getElseEquations() const;

  void setElseEquations(Children<Equation> equations);

private:
  Children<Expression> conditions;
  llvm::SmallVector<Children<Equation>> branches;
  Children<Equation> elseEquations;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_IFEQUATION_H
