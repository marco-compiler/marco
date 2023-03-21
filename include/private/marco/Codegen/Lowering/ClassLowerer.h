#ifndef MARCO_CODEGEN_LOWERING_CLASSBRIDGE_H
#define MARCO_CODEGEN_LOWERING_CLASSBRIDGE_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/Lowering/ModelLowerer.h"
#include "marco/Codegen/Lowering/RecordLowerer.h"
#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"
#include "marco/Codegen/BridgeInterface.h"
#include <memory>

namespace marco::codegen::lowering
{
  class ClassLowerer : public Lowerer
  {
    public:
      ClassLowerer(LoweringContext* context, BridgeInterface* bridge);

      std::vector<mlir::Operation*> operator()(const ast::PartialDerFunction& function);
      std::vector<mlir::Operation*> operator()(const ast::StandardFunction& function);
      std::vector<mlir::Operation*> operator()(const ast::Model& model);
      std::vector<mlir::Operation*> operator()(const ast::Package& package);
      std::vector<mlir::Operation*> operator()(const ast::Record& record);

    private:
      std::unique_ptr<ModelLowerer> modelLowerer;
      std::unique_ptr<RecordLowerer> recordLowerer;
      std::unique_ptr<StandardFunctionLowerer> standardFunctionLowerer;
  };
}

#endif // MARCO_CODEGEN_LOWERING_CLASSBRIDGE_H
