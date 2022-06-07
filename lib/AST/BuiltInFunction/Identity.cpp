#include "marco/AST/BuiltInFunction/Identity.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string IdentityFunction::getName() const
  {
    return "identity";
  }

  std::vector<long> IdentityFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> IdentityFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    // 2D array as output
    return makeType<BuiltInType::Integer>(-1, -1);
  }

  bool IdentityFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void IdentityFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    // Square matrix size
    ranks.push_back(0);
  }
}
