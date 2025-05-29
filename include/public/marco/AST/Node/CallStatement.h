#ifndef PUBLIC_MARCO_AST_NODE_CALLSTATEMENT_H
#define PUBLIC_MARCO_AST_NODE_CALLSTATEMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "marco/AST/Node/Statement.h"
#include "marco/Parser/Location.h"
#include <llvm/Support/JSON.h>
#include <memory>

namespace marco::ast {
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
} // namespace marco::ast

#endif // PUBLIC_MARCO_AST_NODE_CALLSTATEMENT_H
