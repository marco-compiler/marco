#include "marco/AST/BuiltInFunction/Transpose.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string TransposeFunction::getName() const
  {
    return "transpose";
  }

  std::vector<long> TransposeFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> TransposeFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    auto type = args[0]->getType();
    llvm::SmallVector<ArrayDimension, 2> dimensions;

    dimensions.push_back(type[1].isDynamic() ? -1 : type[1].getNumericSize());
    dimensions.push_back(type[0].isDynamic() ? -1 : type[0].getNumericSize());

    type.setDimensions(dimensions);
    return type;
  }

  bool TransposeFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void TransposeFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(2);
  }
}
