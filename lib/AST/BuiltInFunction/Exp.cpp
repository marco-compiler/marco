#include "marco/AST/BuiltInFunction/Exp.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string ExpFunction::getName() const
  {
    return "exp";
  }

  std::vector<long> ExpFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> ExpFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool ExpFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void ExpFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
