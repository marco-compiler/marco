#include "marco/AST/BuiltInFunction/Acos.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string AcosFunction::getName() const
  {
    return "acos";
  }

  std::vector<long> AcosFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> AcosFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool AcosFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void AcosFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
