#ifndef MARCO_AST_BUILTINFUNCTION_SQRT_H
#define MARCO_AST_BUILTINFUNCTION_SQRT_H

#include "marco/AST/BuiltInFunction.h"

namespace marco::ast::builtin
{
  struct SqrtFunction : public BuiltInFunction
  {
    std::string getName() const final;

    std::vector<long> getPossibleArgumentsCount() const final;

    llvm::Optional<Type> resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const final;

    bool canBeCalledElementWise() const final;

    void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const final;
  };
}

#endif // MARCO_AST_BUILTINFUNCTION_SQRT_H
