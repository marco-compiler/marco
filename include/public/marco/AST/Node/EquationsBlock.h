#ifndef MARCO_AST_NODE_EQUATIONSBLOCK_H
#define MARCO_AST_NODE_EQUATIONSBLOCK_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"

namespace marco::ast
{
  class Equation;
  class ForEquation;

  class EquationsBlock : public ASTNode
  {
    public:
      explicit EquationsBlock(SourceRange location);

      EquationsBlock(const EquationsBlock& other);

      ~EquationsBlock() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::EquationsBlock;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      size_t getNumOfEquations() const;

      Equation* getEquation(size_t index);

      const Equation* getEquation(size_t index) const;

      void setEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      void addEquation(std::unique_ptr<ASTNode> node);

      size_t getNumOfForEquations() const;

      ForEquation* getForEquation(size_t index);

      const ForEquation* getForEquation(size_t index) const;

      void setForEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

      void addForEquation(std::unique_ptr<ASTNode> node);

    private:
      llvm::SmallVector<std::unique_ptr<ASTNode>> equations;
      llvm::SmallVector<std::unique_ptr<ASTNode>> forEquations;
  };
}

#endif // MARCO_AST_NODE_EQUATIONSBLOCK_H
