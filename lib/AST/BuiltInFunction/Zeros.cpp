#include "marco/AST/BuiltInFunction/Zeros.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string ZerosFunction::getName() const
  {
    return "zeros";
  }

  std::vector<long> ZerosFunction::getPossibleArgumentsCount() const
  {
    return { kVariadic };
  }

  llvm::Optional<Type> ZerosFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    llvm::SmallVector<ArrayDimension, 2> dimensions(args.size(), -1);
    return Type(BuiltInType::Integer, dimensions);
  }

  bool ZerosFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void ZerosFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    // All the arguments are scalars
    for (size_t i = 0; i < argsCount; ++i) {
      ranks.push_back(0);
    }
  }
}
