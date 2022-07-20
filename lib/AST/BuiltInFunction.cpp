#include "marco/AST/BuiltInFunction.h"
#include "marco/AST/BuiltInFunction/Abs.h"
#include "marco/AST/BuiltInFunction/Acos.h"
#include "marco/AST/BuiltInFunction/Asin.h"
#include "marco/AST/BuiltInFunction/Atan.h"
#include "marco/AST/BuiltInFunction/Atan2.h"
#include "marco/AST/BuiltInFunction/Ceil.h"
#include "marco/AST/BuiltInFunction/Cos.h"
#include "marco/AST/BuiltInFunction/Cosh.h"
#include "marco/AST/BuiltInFunction/Der.h"
#include "marco/AST/BuiltInFunction/Diagonal.h"
#include "marco/AST/BuiltInFunction/Div.h"
#include "marco/AST/BuiltInFunction/Exp.h"
#include "marco/AST/BuiltInFunction/Floor.h"
#include "marco/AST/BuiltInFunction/Identity.h"
#include "marco/AST/BuiltInFunction/Integer.h"
#include "marco/AST/BuiltInFunction/Linspace.h"
#include "marco/AST/BuiltInFunction/Log.h"
#include "marco/AST/BuiltInFunction/Log10.h"
#include "marco/AST/BuiltInFunction/Max.h"
#include "marco/AST/BuiltInFunction/Min.h"
#include "marco/AST/BuiltInFunction/Mod.h"
#include "marco/AST/BuiltInFunction/Ndims.h"
#include "marco/AST/BuiltInFunction/Ones.h"
#include "marco/AST/BuiltInFunction/Product.h"
#include "marco/AST/BuiltInFunction/Rem.h"
#include "marco/AST/BuiltInFunction/Sign.h"
#include "marco/AST/BuiltInFunction/Sin.h"
#include "marco/AST/BuiltInFunction/Sinh.h"
#include "marco/AST/BuiltInFunction/Size.h"
#include "marco/AST/BuiltInFunction/Sqrt.h"
#include "marco/AST/BuiltInFunction/Sum.h"
#include "marco/AST/BuiltInFunction/Symmetric.h"
#include "marco/AST/BuiltInFunction/Tan.h"
#include "marco/AST/BuiltInFunction/Tanh.h"
#include "marco/AST/BuiltInFunction/Transpose.h"
#include "marco/AST/BuiltInFunction/Zeros.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::ast::builtin;

namespace marco::ast
{
  BuiltInFunction::BuiltInFunction() = default;

  BuiltInFunction::BuiltInFunction(const BuiltInFunction& other) = default;

  BuiltInFunction::BuiltInFunction(BuiltInFunction&& other) = default;

  BuiltInFunction& BuiltInFunction::operator=(BuiltInFunction&& other) = default;

  BuiltInFunction::~BuiltInFunction() = default;

  BuiltInFunction& BuiltInFunction::operator=(const BuiltInFunction& other) = default;

  std::vector<std::unique_ptr<BuiltInFunction>> getBuiltInFunctions()
  {
    std::vector<std::unique_ptr<BuiltInFunction>> result;

    result.push_back(std::make_unique<AbsFunction>());
    result.push_back(std::make_unique<AcosFunction>());
    result.push_back(std::make_unique<AsinFunction>());
    result.push_back(std::make_unique<AtanFunction>());
    result.push_back(std::make_unique<Atan2Function>());
    result.push_back(std::make_unique<CeilFunction>());
    result.push_back(std::make_unique<CosFunction>());
    result.push_back(std::make_unique<CoshFunction>());
    result.push_back(std::make_unique<DerFunction>());
    result.push_back(std::make_unique<DiagonalFunction>());
    result.push_back(std::make_unique<DivFunction>());
    result.push_back(std::make_unique<ExpFunction>());
    result.push_back(std::make_unique<FloorFunction>());
    result.push_back(std::make_unique<IdentityFunction>());
    result.push_back(std::make_unique<IntegerFunction>());
    result.push_back(std::make_unique<LinspaceFunction>());
    result.push_back(std::make_unique<LogFunction>());
    result.push_back(std::make_unique<Log10Function>());
    result.push_back(std::make_unique<MaxFunction>());
    result.push_back(std::make_unique<MinFunction>());
    result.push_back(std::make_unique<ModFunction>());
    result.push_back(std::make_unique<NdimsFunction>());
    result.push_back(std::make_unique<OnesFunction>());
    result.push_back(std::make_unique<ProductFunction>());
    result.push_back(std::make_unique<RemFunction>());
    result.push_back(std::make_unique<SignFunction>());
    result.push_back(std::make_unique<SinFunction>());
    result.push_back(std::make_unique<SinhFunction>());
    result.push_back(std::make_unique<SizeFunction>());
    result.push_back(std::make_unique<SqrtFunction>());
    result.push_back(std::make_unique<SumFunction>());
    result.push_back(std::make_unique<SymmetricFunction>());
    result.push_back(std::make_unique<TanFunction>());
    result.push_back(std::make_unique<TanhFunction>());
    result.push_back(std::make_unique<TransposeFunction>());
    result.push_back(std::make_unique<ZerosFunction>());

    // Check that each function name is unique.
    // Just a safety check in order to be resilient to code changes.
    assert(llvm::all_of(result, [&](const auto& foo) {
      return llvm::count_if(result, [&](const auto& bar) {
        return bar->getName() == foo->getName();
      }) == 1;
    }));

    return result;
  }
}
