#include "marco/AST/BuiltInFunction/Linspace.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string LinspaceFunction::getName() const
  {
    return "linspace";
  }

  std::vector<long> LinspaceFunction::getPossibleArgumentsCount() const
  {
    return { 3 };
  }

  llvm::Optional<Type> LinspaceFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    // The result 1D array has a dynamic size, as it depends on the input argument
    return makeType<BuiltInType::Real>(-1);
  }

  bool LinspaceFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void LinspaceFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0); // x1
    ranks.push_back(0); // x2
    ranks.push_back(0); // n
  }
}
