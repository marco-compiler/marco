#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

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

  LoweringContext::LoweringContext(
      mlir::MLIRContext& context, CodegenOptions options)
      : builder(&context),
        options(std::move(options))
  {
    context.loadDialect<mlir::modelica::ModelicaDialect>();
  }

  mlir::OpBuilder& LoweringContext::getBuilder()
  {
    return builder;
  }

  CodegenOptions& LoweringContext::getOptions()
  {
    return options;
  }

  const CodegenOptions& LoweringContext::getOptions() const
  {
    return options;
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
