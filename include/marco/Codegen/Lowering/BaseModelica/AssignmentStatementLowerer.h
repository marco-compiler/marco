#ifndef MARCO_CODEGEN_LOWERING_BASEMODELICA_ASSIGNMENTSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_BASEMODELICA_ASSIGNMENTSTATEMENTLOWERER_H

#include "marco/AST/BaseModelica/AST.h"
#include "marco/Codegen/Lowering/BaseModelica/BridgeInterface.h"
#include "marco/Codegen/Lowering/BaseModelica/Lowerer.h"

namespace marco::codegen::lowering::bmodelica {
class AssignmentStatementLowerer : public Lowerer {
public:
  explicit AssignmentStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool
  lower(const ast::bmodelica::AssignmentStatement &statement) override;

  [[nodiscard]] bool lowerAssignmentToComponentReference(
      mlir::Location assignmentLoc,
      const ast::bmodelica::ComponentReference &destination,
      mlir::Value value) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering::bmodelica

#endif // MARCO_CODEGEN_LOWERING_BASEMODELICA_ASSIGNMENTSTATEMENTLOWERER_H
