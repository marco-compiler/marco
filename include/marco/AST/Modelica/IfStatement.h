#ifndef MARCO_AST_MODELICA_IFSTATEMENT_H
#define MARCO_AST_MODELICA_IFSTATEMENT_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// If-statement.
class IfStatement
    : public NodeImpl<IfStatement, Statement, NodeKind::Statement_If> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  ChildrenRef<Expression> getConditions() const;

  size_t getNumBranches() const;

  /// Get the statements of a branch.
  ChildrenRef<Statement> getBranch(size_t index) const;

  void addBranch(std::unique_ptr<Expression> condition,
                 Children<Statement> statements);

  ChildrenRef<Statement> getElseStatements() const;

  void setElseStatements(Children<Statement> statements);

private:
  Children<Expression> conditions;
  llvm::SmallVector<Children<Statement>> branches;
  Children<Statement> elseStatements;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_IFSTATEMENT_H
