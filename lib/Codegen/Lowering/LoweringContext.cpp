#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen::lowering
{
  LoweringContext::LookupScopeGuard::LookupScopeGuard(LoweringContext* context)
    : context(context), size(context->lookupScopes.size())
  {
    assert(context != nullptr);
  }

  LoweringContext::LookupScopeGuard::~LookupScopeGuard()
  {
    if (context->lookupScopes.size() > size) {
      context->lookupScopes.erase(
          std::next(context->lookupScopes.begin(), size),
          context->lookupScopes.end());
    }
  }

  LoweringContext::LoweringContext(mlir::MLIRContext& context)
      : builder(&context)
  {
    context.loadDialect<mlir::bmodelica::BaseModelicaDialect>();
  }

  mlir::OpBuilder& LoweringContext::getBuilder()
  {
    return builder;
  }

  mlir::SymbolTableCollection& LoweringContext::getSymbolTable()
  {
    return symbolTable;
  }

  LoweringContext::VariablesSymbolTable&
  LoweringContext::getVariablesSymbolTable()
  {
    return variablesSymbolTable;
  }

  mlir::Operation* LoweringContext::getLookupScope()
  {
    assert(!lookupScopes.empty());
    return lookupScopes.back();
  }

  void LoweringContext::pushLookupScope(mlir::Operation* lookupScope)
  {
    lookupScopes.push_back(lookupScope);
  }
}
