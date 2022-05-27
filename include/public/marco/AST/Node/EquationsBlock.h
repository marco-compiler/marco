#ifndef MARCO_AST_NODE_EQUATIONSBLOCK_H
#define MARCO_AST_NODE_EQUATIONSBLOCK_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"

namespace marco::ast
{
  class Equation;
  class ForEquation;

  class EquationsBlock
      : public ASTNode,
        public impl::Cloneable<EquationsBlock>,
        public impl::Dumpable<EquationsBlock>
  {
    private:
      template<typename T> using Container = llvm::SmallVector<T, 3>;

    public:
      EquationsBlock(
        SourceRange location,
        llvm::ArrayRef<std::unique_ptr<Equation>> equations = llvm::None,
        llvm::ArrayRef<std::unique_ptr<ForEquation>> forEquations = llvm::None);

      template<typename... Args>
      static std::unique_ptr<EquationsBlock> build(Args&&... args)
      {
        return std::unique_ptr<EquationsBlock>(new EquationsBlock(std::forward<Args>(args)...));
      }

      EquationsBlock(const EquationsBlock& other);
      EquationsBlock(EquationsBlock&& other);

      ~EquationsBlock() override;

      EquationsBlock& operator=(const EquationsBlock& other);
      EquationsBlock& operator=(EquationsBlock&& other);

      friend void swap(EquationsBlock& first, EquationsBlock& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      llvm::ArrayRef<std::unique_ptr<Equation>> getEquations() const;
      llvm::ArrayRef<std::unique_ptr<ForEquation>> getForEquations() const;

      void add(std::unique_ptr<Equation> equation);
      void add(std::unique_ptr<ForEquation> equation);

    private:
      Container<std::unique_ptr<Equation>> equations;
      Container<std::unique_ptr<ForEquation>> forEquations;
  };
}

#endif // MARCO_AST_NODE_EQUATIONSBLOCK_H
