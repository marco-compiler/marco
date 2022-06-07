#ifndef MARCO_AST_BUILTINFUNCTION_IDENTITY_H
#define MARCO_AST_BUILTINFUNCTION_IDENTITY_H

#include "marco/AST/BuiltInFunction.h"

namespace marco::ast::builtin
{
  struct IdentityFunction : public BuiltInFunction
  {
    std::string getName() const final;

    std::vector<long> getPossibleArgumentsCount() const final;

    llvm::Optional<Type> resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const final;

    bool canBeCalledElementWise() const final;

    void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const final;
  };
}

#endif // MARCO_AST_BUILTINFUNCTION_IDENTITY_H
