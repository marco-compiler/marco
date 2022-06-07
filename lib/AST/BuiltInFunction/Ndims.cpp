#include "marco/AST/BuiltInFunction/Ndims.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string NdimsFunction::getName() const
  {
    return "ndims";
  }

  std::vector<long> NdimsFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> NdimsFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Integer>();
  }

  bool NdimsFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void NdimsFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(-1);
  }
}
