#include "marco/AST/BuiltInFunction/Sum.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string SumFunction::getName() const
  {
    return "sum";
  }

  std::vector<long> SumFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> SumFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return Type(args[0]->getType().get<BuiltInType>());
  }

  bool SumFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void SumFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(-1);
  }
}
