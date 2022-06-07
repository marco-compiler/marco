#include "marco/AST/BuiltInFunction/Log10.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string Log10Function::getName() const
  {
    return "log10";
  }

  std::vector<long> Log10Function::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> Log10Function::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool Log10Function::canBeCalledElementWise() const
  {
    return true;
  }

  void Log10Function::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
