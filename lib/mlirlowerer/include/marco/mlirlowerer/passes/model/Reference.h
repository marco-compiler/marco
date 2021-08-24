#pragma once

#include <mlir/IR/Value.h>

namespace marco::codegen::model
{
	class Reference
	{
		public:
		Reference(mlir::Value var);

		[[nodiscard]] mlir::Value getVar() const;

		private:
		mlir::Value var;
	};
}
