#pragma once

#include <llvm/ADT/SmallVector.h>
#include <marco/mlirlowerer/dialects/modelica/ModelicaBuilder.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include "Attribute.h"
#include "Type.h"

namespace marco::codegen::ida
{
	class IdaBuilder : public modelica::ModelicaBuilder
	{
		public:
		IdaBuilder(mlir::MLIRContext* context);

		[[nodiscard]] OpaquePointerType getOpaquePointerType();
		[[nodiscard]] IntegerPointerType getIntegerPointerType();
		[[nodiscard]] RealPointerType getRealPointerType();

		[[nodiscard]] llvm::SmallVector<mlir::Type, 4> getResidualArgTypes();
		[[nodiscard]] mlir::Type getResidualFunctionType();

		[[nodiscard]] llvm::SmallVector<mlir::Type, 6> getJacobianArgTypes();
		[[nodiscard]] mlir::Type getJacobianFunctionType();
	};
}
