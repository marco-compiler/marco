#ifndef MARCO_CODEN_TRANSFORMS_OUTPUTARRAYSPROMOTION_H
#define MARCO_CODEN_TRANSFORMS_OUTPUTARRAYSPROMOTION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createOutputArraysPromotionPass();

	inline void registerOutputArraysPromotionPass()
	{
		mlir::registerPass(
        "output-arrays-promotion", "Modelica: output arrays promotion",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createOutputArraysPromotionPass();
        });
	}
}

#endif // MARCO_CODEN_TRANSFORMS_OUTPUTARRAYSPROMOTION_H
