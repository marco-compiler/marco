#include "marco/AST/BuiltInFunction/Floor.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string FloorFunction::getName() const
  {
    return "floor";
  }

  std::vector<long> FloorFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> FloorFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool FloorFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void FloorFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
