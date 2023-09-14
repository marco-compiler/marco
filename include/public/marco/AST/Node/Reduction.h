#ifndef MARCO_AST_NODE_REDUCTION_H
#define MARCO_AST_NODE_REDUCTION_H

#include "marco/AST/Node/Expression.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
  class Reduction : public Expression
  {
    public:
      explicit Reduction(SourceRange location);

      Reduction(const Reduction& other);

      ~Reduction() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::Expression_Reduction;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isLValue() const override;

      Expression* getCallee();

      const Expression* getCallee() const;

      void setCallee(std::unique_ptr<ASTNode> node);

      Expression* getExpression();

      const Expression* getExpression() const;

      void setExpression(std::unique_ptr<ASTNode> node);

      size_t getNumOfIterators() const;

      Expression* getIterator(size_t index);

      const Expression* getIterator(size_t index) const;

      llvm::ArrayRef<std::unique_ptr<ASTNode>> getIterators() const;

      void setIterators(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      std::unique_ptr<ASTNode> callee;
      std::unique_ptr<ASTNode> expression;
      llvm::SmallVector<std::unique_ptr<ASTNode>> iterators;
  };
}

#endif // MARCO_AST_NODE_REDUCTION_H
