#ifndef MARCO_CODEGEN_LOWERING_MODELLOWERER_H
#define MARCO_CODEGEN_LOWERING_MODELLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class ModelLowerer : public Lowerer
  {
    public:
      ModelLowerer(BridgeInterface* bridge);

      void declare(const ast::Model& model) override;

      void declareVariables(const ast::Model& model) override;

      void lower(const ast::Model& model) override;

      void lowerVariableAttributes(
          mlir::bmodelica::ModelOp modelOp,
          const ast::Member& variable);

      void lowerVariableAttributes(
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
