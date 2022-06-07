#include "marco/AST/BuiltInFunction/Atan2.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string Atan2Function::getName() const
  {
    return "atan2";
  }

  std::vector<long> Atan2Function::getPossibleArgumentsCount() const
  {
    return { 2 };
  }

  llvm::Optional<Type> Atan2Function::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool Atan2Function::canBeCalledElementWise() const
  {
    return true;
  }

  void Atan2Function::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
    ranks.push_back(0);
  }
}
