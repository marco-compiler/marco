#ifndef MARCO_CODEGEN_LOWERING_MODELLOWERER_H
#define MARCO_CODEGEN_LOWERING_MODELLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ModelLowerer : public Lowerer
  {
    public:
      explicit ModelLowerer(BridgeInterface* bridge);

      void declare(const ast::Model& model) override;

      __attribute__((warn_unused_result)) bool declareVariables(const ast::Model& model) override;

      __attribute__((warn_unused_result)) bool lower(const ast::Model& model) override;

      __attribute__((warn_unused_result)) bool lowerVariableAttributes(
          mlir::bmodelica::ModelOp modelOp,
          const ast::Member& variable);

      __attribute__((warn_unused_result)) bool lowerVariableAttributes(
          mlir::bmodelica::ModelOp modelOp,
          llvm::SmallVectorImpl<mlir::bmodelica::VariableOp>& components,
          const ast::ClassModification& classModification);

    protected:
      using Lowerer::declare;
      using Lowerer::declareVariables;
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_MODELLOWERER_H
