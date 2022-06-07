#include "marco/AST/BuiltInFunction/Tan.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string TanFunction::getName() const
  {
    return "tan";
  }

  std::vector<long> TanFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> TanFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool TanFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void TanFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
