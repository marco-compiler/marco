#ifndef MARCO_CODEGEN_LOWERING_ASSIGNMENTSTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_ASSIGNMENTSTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class AssignmentStatementLowerer : public Lowerer {
public:
  explicit AssignmentStatementLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::AssignmentStatement &statement) override;

  [[nodiscard]] bool lowerAssignmentToComponentReference(
      mlir::Location assignmentLoc, const ast::ComponentReference &destination,
      mlir::Value value) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_ASSIGNMENTSTATEMENTLOWERER_H
