#include "marco/AST/BuiltInFunction/Fill.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string FillFunction::getName() const
  {
    return "fill";
  }

  std::vector<long> FillFunction::getPossibleArgumentsCount() const
  {
    return { kVariadic };
  }

  llvm::Optional<Type> FillFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    llvm::SmallVector<ArrayDimension, 2> dimensions(args.size(), -1);
    return Type(BuiltInType::Real, dimensions);
  }

  bool FillFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void FillFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    for (unsigned int i = 0; i < argsCount; ++i) {
      ranks.push_back(0);
    }
  }
}
