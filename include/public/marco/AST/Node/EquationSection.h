#ifndef MARCO_AST_NODE_EQUATIONSECTION_H
#define MARCO_AST_NODE_EQUATIONSECTION_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/STLExtras.h"
#include <memory>

namespace marco::ast
{
  class Equation;

  class EquationSection : public ASTNode
  {
    public:
      explicit EquationSection(SourceRange location);

      EquationSection(const EquationSection& other);

      ~EquationSection() override;

      static bool classof(const ASTNode* node)
      {
        return node->getKind() == ASTNode::Kind::EquationSection;
      }

      std::unique_ptr<ASTNode> clone() const override;

      llvm::json::Value toJSON() const override;

      bool isInitial() const;

      void setInitial(bool value);

      size_t getNumOfEquations() const;

      bool empty() const;

      Equation* getEquation(size_t index);

      const Equation* getEquation(size_t index) const;

      llvm::ArrayRef<std::unique_ptr<ASTNode>> getEquations();

      void setEquations(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes);

    private:
      bool initial{false};
      llvm::SmallVector<std::unique_ptr<ASTNode>> equations;
  };
}

#endif // MARCO_AST_NODE_EQUATIONSECTION_H
