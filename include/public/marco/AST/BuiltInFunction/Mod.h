#ifndef MARCO_AST_BUILTINFUNTION_MOD_H
#define MARCO_AST_BUILTINFUNTION_MOD_H

#include "marco/AST/BuiltInFunction.h"

namespace marco::ast::builtin
{
  struct ModFunction : public BuiltInFunction
  {
    std::string getName() const final;

    std::vector<long> getPossibleArgumentsCount() const final;

    llvm::Optional<Type> resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const final;

    bool canBeCalledElementWise() const final;

    void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const final;
  };
}

#endif // MARCO_AST_BUILTINFUNTION_MOD_H
