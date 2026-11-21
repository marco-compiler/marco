#ifndef MARCO_AST_MODELICA_WHILESTATEMENT_H
#define MARCO_AST_MODELICA_WHILESTATEMENT_H

#include "marco/AST/Modelica/Expression.h"
#include "marco/AST/Modelica/NodeImpl.h"
#include "marco/AST/Modelica/Statement.h"

namespace marco::ast::modelica {
/// While-statement.
class WhileStatement
    : public NodeImpl<WhileStatement, Statement, NodeKind::Statement_While> {
public:
  using NodeImpl::NodeImpl;

  void walkChildren(llvm::function_ref<void(Node *)> callback) override;

protected:
  void addJSONProperties(llvm::json::Object &obj) const override;

public:
  Expression *getCondition() const;

  void setCondition(std::unique_ptr<Expression> node);

  ChildrenRef<Statement> getStatements() const;

  void setStatements(Children<Statement> body);

private:
  Child<Expression> condition;
  Children<Statement> statements;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_WHILESTATEMENT_H
