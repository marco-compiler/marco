#include "marco/AST/BuiltInFunction/Diagonal.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string DiagonalFunction::getName() const
  {
    return "diagonal";
  }

  std::vector<long> DiagonalFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> DiagonalFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    // 2D array as output
    return makeType<BuiltInType::Integer>(-1, -1);
  }

  bool DiagonalFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void DiagonalFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    // 1D array as input
    ranks.push_back(1);
  }
}
