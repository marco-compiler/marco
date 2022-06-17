#ifndef MARCO_CODEN_TRANSFORMS_EXPLICITCASTINSERTION_H
#define MARCO_CODEN_TRANSFORMS_EXPLICITCASTINSERTION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createExplicitCastInsertionPass();
}

#endif // MARCO_CODEN_TRANSFORMS_EXPLICITCASTINSERTION_H
