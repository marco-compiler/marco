#ifndef MARCO_AST_NODE_FORSTATEMENT_H
#define MARCO_AST_NODE_FORSTATEMENT_H

#include "marco/AST/Node/Statement.h"

namespace marco::ast
{
  class Induction;

  class ForStatement : public Statement
  {
    public:
      explicit ForStatement(SourceRange location);

      ForStatement(const ForStatement& other);

      ~ForStatement() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Statement_For;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      Induction* getInduction();

      const Induction* getInduction() const;

      void setInduction(std::unique_ptr<ASTNode> node);

      size_t size() const;

      Statement* operator[](size_t index);

      const Statement* operator[](size_t index) const;

      llvm::ArrayRef<std::unique_ptr<ASTNode>> getStatements() const;

      void setStatements(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      std::unique_ptr<ASTNode> induction;
      llvm::SmallVector<std::unique_ptr<ASTNode>> statements;
  };
}

#endif // MARCO_AST_NODE_FORSTATEMENT_H
