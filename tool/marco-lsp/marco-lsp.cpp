#include "marco/Dialect/BaseModelica/IR/Dialect.h"
#include "marco/Dialect/IDA/IR/Dialect.h"
#include "marco/Dialect/KINSOL/IR/Dialect.h"
#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "marco/Dialect/Modeling/IR/Dialect.h"
#include "marco/Dialect/Runtime/IR/Dialect.h"
#include "marco/Dialect/SUNDIALS/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  registry
      .insert<mlir::bmodelica::BaseModelicaDialect, mlir::ida::IDADialect,
              mlir::kinsol::KINSOLDialect, mlir::modelica::ModelicaDialect,
              mlir::modeling::ModelingDialect, mlir::runtime::RuntimeDialect,
              mlir::sundials::SUNDIALSDialect>();

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
