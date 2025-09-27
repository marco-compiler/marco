#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_ARRAYGENERATORLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_ARRAYGENERATORLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class ArrayGeneratorLowerer : public Lowerer {
public:
  explicit ArrayGeneratorLowerer(BridgeInterface *bridge);

  std::optional<Results>
  lower(const ast::bmodelica::ArrayGenerator &array) override;

protected:
  using Lowerer::lower;

private:
  std::optional<Results> lower(const ast::bmodelica::ArrayConstant &array);

  std::optional<Results> lower(const ast::bmodelica::ArrayForGenerator &array);

  void computeShape(const ast::bmodelica::ArrayGenerator &array,
                    llvm::SmallVectorImpl<int64_t> &outShape);

  [[nodiscard]] bool lowerValues(const ast::bmodelica::Expression &array,
                                 llvm::SmallVectorImpl<mlir::Value> &outValues);
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_ARRAYGENERATORLOWERER_H
