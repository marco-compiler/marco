#pragma once

#include <mlir/IR/Value.h>

namespace modelica::codegen::model
{
	class Constant
	{
		public:
		Constant(mlir::Value value);

		[[nodiscard]] size_t childrenCount() const;

		private:
		mlir::Value value;
	};
}
