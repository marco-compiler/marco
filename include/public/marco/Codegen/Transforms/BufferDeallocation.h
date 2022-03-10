#ifndef MARCO_CODEN_TRANSFORMS_BUFFERDEALLOCATION_H
#define MARCO_CODEN_TRANSFORMS_BUFFERDEALLOCATION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	std::unique_ptr<mlir::Pass> createBufferDeallocationPass();

	inline void registerBufferDeallocationPass()
	{
		mlir::registerPass(
        "buffer-deallocation", "Modelica: automatic buffer deallocation",
        []() -> std::unique_ptr<::mlir::Pass> {
          return createBufferDeallocationPass();
        });
	}
}

#endif // MARCO_CODEN_TRANSFORMS_BUFFERDEALLOCATION_H
