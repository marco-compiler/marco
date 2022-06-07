#include "marco/AST/BuiltInFunction/Abs.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string AbsFunction::getName() const
  {
    return "abs";
  }

  std::vector<long> AbsFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> AbsFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return Type(args[0]->getType().get<BuiltInType>());
  }

  bool AbsFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void AbsFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
