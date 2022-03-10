#ifndef MARCO_CODEN_TRANSFORMS_EXPLICITCASTINSERTION_H
#define MARCO_CODEN_TRANSFORMS_EXPLICITCASTINSERTION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();

	inline void registerExplicitCastInsertionPass()
	{
		mlir::registerPass(
        "explicit-cast-insertion", "Modelica: explicit cast insertion",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createExplicitCastInsertionPass();
        });
	}
}

#endif // MARCO_CODEN_TRANSFORMS_EXPLICITCASTINSERTION_H
