#pragma once

#include <mlir/IR/Value.h>

namespace modelica::codegen::model
{
	class Constant
	{
		public:
		Constant(mlir::Value value);

		private:
		mlir::Value value;
	};
}
