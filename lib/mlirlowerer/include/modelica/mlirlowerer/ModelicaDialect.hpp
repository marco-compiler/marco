#pragma once

#include <mlir/IR/Dialect.h>

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
