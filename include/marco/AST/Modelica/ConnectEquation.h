#ifndef MARCO_AST_MODELICA_CONNECTEQUATION_H
#define MARCO_AST_MODELICA_CONNECTEQUATION_H

#include "marco/AST/Modelica/ComponentReference.h"
#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Connect-equation.
class ConnectEquation
    : public NodeImpl<ConnectEquation, Equation, NodeKind::Equation_Connect> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ComponentReference *getLhs() const;

  void setLhs(std::unique_ptr<ComponentReference> node);

  ComponentReference *getRhs() const;

  void setRhs(std::unique_ptr<ComponentReference> node);

private:
  Child<ComponentReference> lhs;
  Child<ComponentReference> rhs;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_CONNECTEQUATION_H
