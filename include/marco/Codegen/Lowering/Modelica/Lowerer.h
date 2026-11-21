#ifndef MARCO_CODEGEN_LOWERING_MODELICA_LOWERER_H
#define MARCO_CODEGEN_LOWERING_MODELICA_LOWERER_H

#include "marco/AST/Modelica/AST.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>

namespace marco::codegen::lowering::modelica {
class LowererImpl;

/// Lowering of the Modelica AST to the class-level operations of the Modelica
/// dialect. The files are merged into a single class tree, and all its classes
/// are lowered. The names are kept as written in the source code, while the
/// types of the values are determined by looking up the declarations of the
/// referenced elements.
class Lowerer {
public:
  explicit Lowerer(mlir::MLIRContext &context);

  ~Lowerer();

  /// Add a parsed file. Files with a within-clause must be added after the
  /// file defining the enclosing package. The AST must outlive the lowerer.
  mlir::LogicalResult addFile(const ast::modelica::StoredDefinition &file);

  /// Lower all the classes into the given module. The classes whose lowering
  /// fails are reported through the diagnostics of the context and omitted
  /// from the module, and failure is returned.
  mlir::LogicalResult lower(mlir::ModuleOp module);

private:
  std::unique_ptr<LowererImpl> impl;
};
} // namespace marco::codegen::lowering::modelica

#endif // MARCO_CODEGEN_LOWERING_MODELICA_LOWERER_H
