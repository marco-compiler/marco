#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SOLVERS_SUNDIALS_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SOLVERS_SUNDIALS_H

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/SUNDIALS/IR/SUNDIALS.h"

namespace mlir::bmodelica {
class PartialDerivativeTemplatesCollection {
  struct Info {
    /// The function implementing the generic partial derivative function, which
    /// is independent from the actual indices of the equation.
    FunctionOp funcOp;

    /// The position of variables among function arguments.
    llvm::MapVector<VariableOp, size_t> variablesPos;
  };

  /// Keep track of the equation templates for which a partial derivative
  /// function has already been created.
  llvm::DenseMap<EquationTemplateOp, Info> info;

public:
  size_t size() const;

  bool hasEquationTemplate(EquationTemplateOp equationTemplateOp) const;

  std::optional<FunctionOp>
  getDerivativeTemplate(EquationTemplateOp equationTemplateOp) const;

  size_t getVariablesCount(EquationTemplateOp equationTemplateOp) const;

  llvm::SetVector<VariableOp>
  getVariables(EquationTemplateOp equationTemplateOp) const;

  std::optional<size_t> getVariablePos(EquationTemplateOp equationTemplateOp,
                                       VariableOp variableOp) const;

  void setDerivativeTemplate(EquationTemplateOp equationTemplateOp,
                             FunctionOp derTemplateFuncOp);

  void setVariablePos(EquationTemplateOp equationTemplateOp,
                      VariableOp variableOp, size_t pos);
};

mlir::sundials::VariableGetterOp
createGetterFunction(mlir::OpBuilder &builder,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     mlir::Location loc, mlir::ModuleOp moduleOp,
                     VariableOp variable, llvm::StringRef functionName);

mlir::sundials::VariableGetterOp
createGetterFunction(mlir::OpBuilder &builder,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     mlir::Location loc, mlir::ModuleOp moduleOp,
                     GlobalVariableOp variable, llvm::StringRef functionName);

mlir::sundials::VariableSetterOp
createSetterFunction(mlir::OpBuilder &builder,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     mlir::Location loc, mlir::ModuleOp moduleOp,
                     VariableOp variable, llvm::StringRef functionName);

mlir::sundials::VariableSetterOp
createSetterFunction(mlir::OpBuilder &builder,
                     mlir::SymbolTableCollection &symbolTableCollection,
                     mlir::Location loc, mlir::ModuleOp moduleOp,
                     GlobalVariableOp variable, llvm::StringRef functionName);

GlobalVariableOp createGlobalADSeed(mlir::OpBuilder &builder,
                                    mlir::ModuleOp moduleOp, mlir::Location loc,
                                    llvm::StringRef name, mlir::Type type);

void setGlobalADSeed(mlir::OpBuilder &builder, mlir::Location loc,
                     GlobalVariableOp seedVariableOp, mlir::ValueRange indices,
                     mlir::Value value);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_SOLVERS_SUNDIALS_H
