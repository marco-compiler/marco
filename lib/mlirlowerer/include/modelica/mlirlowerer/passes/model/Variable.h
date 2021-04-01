#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Value.h>
#include <modelica/utils/IndexSet.hpp>

namespace modelica::codegen::model
{
	class Variable
	{
		public:
		Variable(mlir::Value memory);

		mlir::Value getReference();

		[[nodiscard]] bool isState() const;
		[[nodiscard]] bool isConstant() const;

		mlir::Value getDer();
		void setDer(mlir::Value value);

		[[nodiscard]] IndexSet toIndexSet() const;
		[[nodiscard]] MultiDimInterval toMultiDimInterval() const;

		[[nodiscard]] size_t indexOfElement(llvm::ArrayRef<size_t> access) const;

		private:
		mlir::Value reference;
		bool state;
		bool constant;
		mlir::Value der;
	};
}