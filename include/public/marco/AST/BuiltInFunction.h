#ifndef MARCO_AST_BUILTINFUNCTION_BUILTINFUNCTION_H
#define MARCO_AST_BUILTINFUNCTION_BUILTINFUNCTION_H

#include "marco/AST/AST.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace marco::ast
{
  struct BuiltInFunction
  {
    static constexpr long kVariadic = -1;

    BuiltInFunction();
    BuiltInFunction(const BuiltInFunction& other);

    BuiltInFunction(BuiltInFunction&& other);
    BuiltInFunction& operator=(BuiltInFunction&& other);

    virtual ~BuiltInFunction();

    BuiltInFunction& operator=(const BuiltInFunction& other);

    virtual std::string getName() const = 0;

    /// Get a list of the possible amounts of arguments.
    virtual std::vector<long> getPossibleArgumentsCount() const = 0;

    /// Get the result type in case of non element-wise call.
    /// The arguments str needed because some functions (such
    /// as min / max / size) may vary their behaviour according to it.
    /// If the arguments count is invalid, then no type is returned.
    virtual llvm::Optional<Type> resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const = 0;

    /// Whether the function can be used in an element-wise call.
    virtual bool canBeCalledElementWise() const = 0;

    /// Get the expected rank for each argument of the function.
    /// A rank of -1 means any rank is accepted (functions like ndims are
    /// made for the exact purpose to receive such arguments).
    /// The total arguments count is needed because some functions (such
    /// as min / max) may vary their behaviour according to it.
    virtual void getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const = 0;
  };

  /// Get all the built-in functions
  std::vector<std::unique_ptr<BuiltInFunction>> getBuiltInFunctions();
}

#endif // MARCO_AST_BUILTINFUNCTION_BUILTINFUNCTION_H
