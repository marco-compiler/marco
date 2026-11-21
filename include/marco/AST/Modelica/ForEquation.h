#ifndef MARCO_AST_MODELICA_FOREQUATION_H
#define MARCO_AST_MODELICA_FOREQUATION_H

#include "marco/AST/Modelica/Equation.h"
#include "marco/AST/Modelica/ForIndex.h"
#include "marco/AST/Modelica/NodeImpl.h"

namespace marco::ast::modelica {
/// For-equation.
class ForEquation
    : public NodeImpl<ForEquation, Equation, NodeKind::Equation_For> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ChildrenRef<ForIndex> getIndices() const;

  void addIndex(std::unique_ptr<ForIndex> node);

  ChildrenRef<Equation> getEquations() const;

  void setEquations(Children<Equation> body);

private:
  Children<ForIndex> indices;
  Children<Equation> equations;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_FOREQUATION_H
