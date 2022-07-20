#include "marco/AST/BuiltInFunction/Integer.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string IntegerFunction::getName() const
  {
    return "integer";
  }

  std::vector<long> IntegerFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> IntegerFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Integer>();
  }

  bool IntegerFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void IntegerFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
