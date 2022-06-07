#include "marco/AST/BuiltInFunction/Ones.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string OnesFunction::getName() const
  {
    return "ones";
  }

  std::vector<long> OnesFunction::getPossibleArgumentsCount() const
  {
    return { kVariadic };
  }

  llvm::Optional<Type> OnesFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    llvm::SmallVector<ArrayDimension, 2> dimensions(args.size(), -1);
    return Type(BuiltInType::Integer, dimensions);
  }

  bool OnesFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void OnesFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    // All the arguments are scalars
    for (size_t i = 0; i < argsCount; ++i) {
      ranks.push_back(0);
    }
  }
}
