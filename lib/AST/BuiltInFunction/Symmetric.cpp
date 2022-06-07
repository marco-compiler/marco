#include "marco/AST/BuiltInFunction/Symmetric.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string SymmetricFunction::getName() const
  {
    return "symmetric";
  }

  std::vector<long> SymmetricFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> SymmetricFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return args[0]->getType();
  }

  bool SymmetricFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void SymmetricFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(2);
  }
}
