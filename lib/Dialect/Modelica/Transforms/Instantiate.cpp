#include "marco/Dialect/Modelica/Transforms/Instantiate.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "marco/Dialect/Modelica/Transforms/Instantiation/Instantiator.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir::modelica {
#define GEN_PASS_DEF_INSTANTIATEPASS
#include "marco/Dialect/Modelica/Transforms/Passes.h.inc"
} // namespace mlir::modelica

using namespace ::mlir::modelica;

namespace {
class InstantiatePass final
    : public mlir::modelica::impl::InstantiatePassBase<InstantiatePass> {
public:
  using InstantiatePassBase<InstantiatePass>::InstantiatePassBase;

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();

    if (modelName.empty()) {
      moduleOp.emitError("the name of the model to be instantiated is "
                         "missing");
      return signalPassFailure();
    }

    // Errors may be reported also by the lookups that don't stop the
    // instantiation.
    bool hasErrors = false;

    mlir::ScopedDiagnosticHandler handler(
        &getContext(), [&](mlir::Diagnostic &diagnostic) {
          if (diagnostic.getSeverity() == mlir::DiagnosticSeverity::Error) {
            hasErrors = true;
          }

          return mlir::failure();
        });

    instantiation::Instantiator instantiator(moduleOp);

    if (mlir::failed(instantiator.instantiateModel(modelName)) || hasErrors) {
      return signalPassFailure();
    }

    instantiator.eraseClasses();
  }
};
} // namespace
