#ifndef MARCO_AST_MODELICA_WHENEQUATION_H
#define MARCO_AST_MODELICA_WHENEQUATION_H

#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// When-equation. The first branch is the `when` one,
/// the others are the `elsewhen` ones.
class WhenEquation
    : public NodeImpl<WhenEquation, Equation, NodeKind::Equation_When> {
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

private:
  Children<Expression> conditions;
  llvm::SmallVector<Children<Equation>> branches;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_WHENEQUATION_H
