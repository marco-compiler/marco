#pragma once

#include <mlir/IR/Dialect.h>

namespace modelica
{
	class ModelicaDialect //: public mlir::Dialect
	{
		public:
		//ModelicaDialect(mlir::MLIRContext* context);

		static llvm::StringRef getDialectNamespace();

	};
}
