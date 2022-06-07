#include "marco/AST/BuiltInFunction/Atan.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string AtanFunction::getName() const
  {
    return "atan";
  }

  std::vector<long> AtanFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> AtanFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool AtanFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void AtanFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
