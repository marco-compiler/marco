#pragma once

#include <mlir/IR/Dialect.h>

#include "Attribute.h"
#include "Ops.h"
#include "Type.h"

namespace modelica
{
	class ModelicaDialect : public mlir::Dialect
	{
		public:
		explicit ModelicaDialect(mlir::MLIRContext* context);

		static llvm::StringRef getDialectNamespace();
		void printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const override;
		void printAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer) const override;
		mlir::Operation* materializeConstant(mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc) override;
	};
}
