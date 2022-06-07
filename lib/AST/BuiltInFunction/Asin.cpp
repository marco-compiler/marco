#include "marco/AST/BuiltInFunction/Asin.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string AsinFunction::getName() const
  {
    return "asin";
  }

  std::vector<long> AsinFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> AsinFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool AsinFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void AsinFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
