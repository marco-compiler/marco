#ifndef MARCO_VARIABLEFILTER_AST_H
#define MARCO_VARIABLEFILTER_AST_H

#include "marco/VariableFilter/Range.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace marco::vf
{
  class ASTNode
  {
    public:
      virtual ~ASTNode();
  };

  class VariableExpression : public ASTNode
  {
    public:
      VariableExpression(llvm::StringRef identifier);

      llvm::StringRef getIdentifier() const;

    private:
      std::string identifier;
  };

  class ArrayExpression : public ASTNode
  {
    public:
      ArrayExpression(VariableExpression variable, llvm::ArrayRef<Range> ranges);

      VariableExpression getVariable() const;

      llvm::ArrayRef<Range> getRanges() const;

    private:
      VariableExpression variable;
      llvm::SmallVector<Range> ranges;
  };

  class DerivativeExpression : public ASTNode
  {
    public:
      DerivativeExpression(VariableExpression derivedVariable);

      VariableExpression getDerivedVariable() const;

    private:
      VariableExpression derivedVariable;
  };

  class RegexExpression : public ASTNode
  {
    public:
      RegexExpression(llvm::StringRef regex);

      llvm::StringRef getRegex() const;

    private:
      std::string regex;
  };
}

#endif // MARCO_VARIABLEFILTER_AST_H
