#include "marco/AST/BuiltInFunction/Mod.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string ModFunction::getName() const
  {
    return "mod";
  }

  std::vector<long> ModFunction::getPossibleArgumentsCount() const
  {
    return { 2 };
  }

  llvm::Optional<Type> ModFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    auto& xType = args[0]->getType();
    auto& yType = args[1]->getType();

    if (xType.isa<BuiltInType::Real>() || yType.isa<BuiltInType::Real>()) {
      return makeType<BuiltInType::Real>();
    }

    return makeType<BuiltInType::Integer>();
  }

  bool ModFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void ModFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
    ranks.push_back(0);
  }
}
