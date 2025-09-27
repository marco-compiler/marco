#ifndef MARCO_AST_BASEMODELICA_NODE_CALLSTATEMENT_H
#define MARCO_AST_BASEMODELICA_NODE_CALLSTATEMENT_H

#include "marco/AST/BaseModelica/Node/Statement.h"

namespace marco::ast::bmodelica {
class Call;

class CallStatement : public Statement {
public:
  explicit CallStatement(SourceRange location);

  CallStatement(const CallStatement &other);

  ~CallStatement() override = default;

  static bool classof(const ASTNode *node) {
    return node->getKind() == ASTNode::Kind::Statement_Call;
  }

  std::unique_ptr<ASTNode> clone() const override;

  llvm::json::Value toJSON() const override;

  Call *getCall();

  const Call *getCall() const;

  void setCall(std::unique_ptr<ASTNode> node);

private:
  // The function call expression
  std::unique_ptr<ASTNode> call;
};
} // namespace marco::ast::bmodelica

#endif // MARCO_AST_BASEMODELICA_NODE_CALLSTATEMENT_H
