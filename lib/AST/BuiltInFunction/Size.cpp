#include "marco/AST/BuiltInFunction/Size.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string SizeFunction::getName() const
  {
    return "size";
  }

  std::vector<long> SizeFunction::getPossibleArgumentsCount() const
  {
    return { 1, 2 };
  }

  llvm::Optional<Type> SizeFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    if (args.size() == 1) {
      return makeType<BuiltInType::Integer>(args[0]->getType().getDimensions().size());
    }

    if (args.size() == 2) {
      return makeType<BuiltInType::Integer>();
    }

    return llvm::None;
  }

  bool SizeFunction::canBeCalledElementWise() const
  {
    return false;
  }

  void SizeFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(-1);

    if (argsCount == 2) {
      ranks.push_back(0);
    }
  }
}
