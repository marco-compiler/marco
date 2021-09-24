#pragma once

#include <mlir/IR/Value.h>

namespace marco::codegen::model
{
	class Induction
	{
		public:
		Induction(mlir::BlockArgument argument);

		[[nodiscard]] mlir::BlockArgument getArgument() const;

		private:
		mlir::BlockArgument argument;
	};
}
