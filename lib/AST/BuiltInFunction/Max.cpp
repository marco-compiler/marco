#include "marco/AST/BuiltInFunction/Max.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string MaxFunction::getName() const
  {
    return "max";
  }

  std::vector<long> MaxFunction::getPossibleArgumentsCount() const
  {
    return { 1, 2 };
  }

  llvm::Optional<Type> MaxFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    if (args.size() == 1) {
      return Type(args[0]->getType().get<BuiltInType>());
    }

    if (args.size() == 2) {
      auto& xType = args[0]->getType();
      auto& yType = args[1]->getType();

      return Type(getMostGenericBuiltInType(xType.get<BuiltInType>(), yType.get<BuiltInType>()));
    }

    return llvm::None;
  }

  bool MaxFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void MaxFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    if (argsCount == 1) {
      // The array can have any rank
      ranks.push_back(-1);
      return;
    }

    if (argsCount == 2) {
      ranks.push_back(0);
      ranks.push_back(0);
      return;
    }

    llvm_unreachable("Unknown number of arguments");
  }
}
