#include "marco/AST/BuiltInFunction/Der.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string DerFunction::getName() const
  {
    return "der";
  }

  std::vector<long> DerFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> DerFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool DerFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void DerFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
