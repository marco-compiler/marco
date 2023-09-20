#ifndef MARCO_AST_NODE_SUBSCRIPT_H
#define MARCO_AST_NODE_SUBSCRIPT_H

#include "marco/AST/Node/ASTNode.h"

namespace marco::ast
{
  class Expression;

  class Subscript : public ASTNode
  {
    public:
      Subscript(SourceRange location);

      Subscript(const Subscript& other);

      ~Subscript() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Subscript;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isUnbounded() const;

      Expression* getExpression();

      const Expression* getExpression() const;

      void setExpression(std::unique_ptr<ASTNode> node);

    private:
      std::unique_ptr<ASTNode> expression;
  };
}

#endif // MARCO_AST_NODE_SUBSCRIPT_H
