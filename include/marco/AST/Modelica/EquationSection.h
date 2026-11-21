#ifndef MARCO_AST_MODELICA_EQUATIONSECTION_H
#define MARCO_AST_MODELICA_EQUATIONSECTION_H

#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// Equation section.
class EquationSection
    : public NodeImpl<EquationSection, Node, NodeKind::EquationSection> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isInitial() const;

  void setInitial(bool value);

  ChildrenRef<Equation> getEquations() const;

  void addEquation(std::unique_ptr<Equation> node);

private:
  bool initial = false;
  Children<Equation> equations;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_EQUATIONSECTION_H
