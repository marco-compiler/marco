#ifndef MARCO_CODEGEN_LOWERING_EXTERNALREFLOWERER_H
#define MARCO_CODEGEN_LOWERING_EXTERNALREFLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/BridgeInterface.h"
#include "marco/Codegen/Lowering/Lowerer.h"

namespace marco::codegen::lowering {
class External_RefLowerer : public Lowerer {
public:
  explicit External_RefLowerer(BridgeInterface *bridge);

  [[nodiscard]] bool lower(const ast::External_Ref &er) override;

protected:
  using Lowerer::lower;
};
} // namespace marco::codegen::lowering

#endif // MARCO_CODEGEN_LOWERING_EXTERNALREFLOWERER_H
