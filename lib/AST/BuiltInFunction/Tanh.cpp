#include "marco/AST/BuiltInFunction/Tanh.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string TanhFunction::getName() const
  {
    return "tanh";
  }

  std::vector<long> TanhFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> TanhFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool TanhFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void TanhFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
