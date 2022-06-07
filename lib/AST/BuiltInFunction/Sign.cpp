#include "marco/AST/BuiltInFunction/Sign.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string SignFunction::getName() const
  {
    return "sign";
  }

  std::vector<long> SignFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> SignFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Integer>();
  }

  bool SignFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void SignFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
