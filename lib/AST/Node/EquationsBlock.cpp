#include "marco/AST/Node/EquationsBlock.h"
#include "marco/AST/Node/Equation.h"
#include "marco/AST/Node/ForEquation.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  EquationsBlock::EquationsBlock(
      SourceRange location,
      llvm::ArrayRef<std::unique_ptr<Equation>> equations,
      llvm::ArrayRef<std::unique_ptr<ForEquation>> forEquations)
    : ASTNode(std::move(location))
  {
    for (const auto& equation : equations) {
      this->equations.push_back(equation->clone());
    }

    for (const auto& forEquation : forEquations) {
      this->forEquations.push_back(forEquation->clone());
    }
  }

  EquationsBlock::EquationsBlock(const EquationsBlock& other)
    : ASTNode(other)
  {
    for (const auto& equation : other.equations) {
      this->equations.push_back(equation->clone());
    }

    for (const auto& forEquation : other.forEquations) {
      this->forEquations.push_back(forEquation->clone());
    }
  }

  EquationsBlock::EquationsBlock(EquationsBlock&& other) = default;

  EquationsBlock::~EquationsBlock() = default;

  EquationsBlock& EquationsBlock::operator=(const EquationsBlock& other)
  {
    EquationsBlock result(other);
    swap(*this, result);
    return *this;
  }

  EquationsBlock& EquationsBlock::operator=(EquationsBlock&& other) = default;

  void swap(EquationsBlock& first, EquationsBlock& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    impl::swap(first.equations, second.equations);
    impl::swap(first.forEquations, second.forEquations);
  }

  void EquationsBlock::print(llvm::raw_ostream& os, size_t indents) const
  {
    for (const auto& equation : equations) {
      equation->dump(os, indents);
    }

    for (const auto& forEquation : forEquations) {
      forEquation->dump(os, indents);
    }
  }

  llvm::ArrayRef<std::unique_ptr<Equation>> EquationsBlock::getEquations() const
  {
    return equations;
  }

  llvm::ArrayRef<std::unique_ptr<ForEquation>> EquationsBlock::getForEquations() const
  {
    return forEquations;
  }

  void EquationsBlock::add(std::unique_ptr<Equation> equation)
  {
    equations.push_back(std::move(equation));
  }

  void EquationsBlock::add(std::unique_ptr<ForEquation> equation)
  {
    forEquations.push_back(std::move(equation));
  }
}
