#pragma once

#include <mlir/IR/Builders.h>

namespace modelica
{
	enum class MlirType
	{
		Index,
		I1,
		I32,
		I64,
		F16,
		F32,
		F64
	};

	template<MlirType From, MlirType To>
	mlir::Value cast(mlir::OpBuilder builder, mlir::Value value)
	{

	}




}
