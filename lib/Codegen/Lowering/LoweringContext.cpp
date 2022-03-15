#include "marco/Codegen/Lowering/LoweringContext.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen::lowering
{
  LoweringContext::LoweringContext(mlir::MLIRContext& context, CodegenOptions options)
      : builder(&context),
        options(std::move(options))
  {
    context.loadDialect<mlir::modelica::ModelicaDialect>();
    context.loadDialect<mlir::StandardOpsDialect>();
  }
}
