#include "marco/AST/BuiltInFunction/Sqrt.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string SqrtFunction::getName() const
  {
    return "sqrt";
  }

  std::vector<long> SqrtFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> SqrtFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool SqrtFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void SqrtFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
