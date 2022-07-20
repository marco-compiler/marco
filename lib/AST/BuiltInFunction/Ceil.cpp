#include "marco/AST/BuiltInFunction/Ceil.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string CeilFunction::getName() const
  {
    return "ceil";
  }

  std::vector<long> CeilFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> CeilFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool CeilFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void CeilFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
