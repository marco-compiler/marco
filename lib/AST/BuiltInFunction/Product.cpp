#include "marco/AST/BuiltInFunction/Product.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string ProductFunction::getName() const
  {
    return "product";
  }

  std::vector<long> ProductFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> ProductFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return Type(args[0]->getType().get<BuiltInType>());
  }

  bool ProductFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void ProductFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(-1);
  }
}
