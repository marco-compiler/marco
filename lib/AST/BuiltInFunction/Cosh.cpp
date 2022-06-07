#include "marco/AST/BuiltInFunction/Cosh.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string CoshFunction::getName() const
  {
    return "cosh";
  }

  std::vector<long> CoshFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> CoshFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool CoshFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void CoshFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
