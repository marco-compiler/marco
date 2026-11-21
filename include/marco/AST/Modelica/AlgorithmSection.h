#ifndef MARCO_AST_MODELICA_ALGORITHMSECTION_H
#define MARCO_AST_MODELICA_ALGORITHMSECTION_H

#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// Algorithm section.
class AlgorithmSection
    : public NodeImpl<AlgorithmSection, Node, NodeKind::AlgorithmSection> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  bool isInitial() const;

  void setInitial(bool value);

  ChildrenRef<Statement> getStatements() const;

  void addStatement(std::unique_ptr<Statement> node);

private:
  bool initial = false;
  Children<Statement> statements;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ALGORITHMSECTION_H
