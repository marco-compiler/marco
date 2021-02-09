#pragma once

#include <mlir/IR/Dialect.h>
#include <modelica/mlirlowerer/Ops.h>
#include <modelica/mlirlowerer/ArrayType.h>

namespace modelica
{
	class ModelicaDialect : public mlir::Dialect
	{
		public:
		explicit ModelicaDialect(mlir::MLIRContext* context);

		/**
		 * Get the dialect namespace.
		 *
		 * @return dialect namespace
		 */
		static llvm::StringRef getDialectNamespace();
	};
}
