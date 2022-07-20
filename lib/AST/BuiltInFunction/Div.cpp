#include "marco/AST/BuiltInFunction/Div.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string DivFunction::getName() const
  {
    return "div";
  }

  std::vector<long> DivFunction::getPossibleArgumentsCount() const
  {
    return { 2 };
  }

  llvm::Optional<Type> DivFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    auto& xType = args[0]->getType();
    auto& yType = args[1]->getType();

    if (xType.isa<BuiltInType::Real>() || yType.isa<BuiltInType::Real>()) {
      return makeType<BuiltInType::Real>();
    }

    return makeType<BuiltInType::Integer>();
  }

  bool DivFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void DivFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
    ranks.push_back(0);
  }
}
