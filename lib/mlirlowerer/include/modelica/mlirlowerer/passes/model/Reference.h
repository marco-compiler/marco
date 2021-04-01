#pragma once

#include <mlir/IR/Value.h>

namespace modelica::codegen::model
{
	class Reference
	{
		public:
		Reference(mlir::Value var);

		[[nodiscard]] mlir::Value getVar() const;

		[[nodiscard]] size_t childrenCount() const;

		private:
		mlir::Value var;
	};
}