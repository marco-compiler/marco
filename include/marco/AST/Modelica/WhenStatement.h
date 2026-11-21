#ifndef MARCO_AST_MODELICA_WHENSTATEMENT_H
#define MARCO_AST_MODELICA_WHENSTATEMENT_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// When-statement.
class WhenStatement
    : public NodeImpl<WhenStatement, Statement, NodeKind::Statement_When> {
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

private:
  Children<Expression> conditions;
  llvm::SmallVector<Children<Statement>> branches;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_WHENSTATEMENT_H
