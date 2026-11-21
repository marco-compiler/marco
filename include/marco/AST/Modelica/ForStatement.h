#ifndef MARCO_AST_MODELICA_FORSTATEMENT_H
#define MARCO_AST_MODELICA_FORSTATEMENT_H

#include "marco/AST/Modelica/ForIndex.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// For-statement.
class ForStatement
    : public NodeImpl<ForStatement, Statement, NodeKind::Statement_For> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ChildrenRef<ForIndex> getIndices() const;

  void addIndex(std::unique_ptr<ForIndex> node);

  ChildrenRef<Statement> getStatements() const;

  void setStatements(Children<Statement> body);

private:
  Children<ForIndex> indices;
  Children<Statement> statements;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_FORSTATEMENT_H
