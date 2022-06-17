#ifndef MARCO_CODEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
#define MARCO_CODEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createAutomaticDifferentiationPass();
}

#endif // MARCO_CODEN_TRANSFORMS_AUTOMATICDIFFERENTIATION_H
