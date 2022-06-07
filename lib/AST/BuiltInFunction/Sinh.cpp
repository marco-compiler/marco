#include "marco/AST/BuiltInFunction/Sinh.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast::builtin
{
  std::string SinhFunction::getName() const
  {
    return "sinh";
  }

  std::vector<long> SinhFunction::getPossibleArgumentsCount() const
  {
    return { 1 };
  }

  llvm::Optional<Type> SinhFunction::resultType(llvm::ArrayRef<std::unique_ptr<Expression>> args) const
  {
    return makeType<BuiltInType::Real>();
  }

  bool SinhFunction::canBeCalledElementWise() const
  {
    return true;
  }

  void SinhFunction::getArgsExpectedRanks(unsigned int argsCount, llvm::SmallVectorImpl<long>& ranks) const
  {
    ranks.push_back(0);
  }
}
