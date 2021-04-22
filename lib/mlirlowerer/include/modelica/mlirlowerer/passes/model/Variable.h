#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <memory>
#include <mlir/IR/Value.h>
#include <modelica/utils/IndexSet.hpp>

namespace modelica::codegen::model
{
	class Variable
	{
		private:
		class Impl;

		std::shared_ptr<Impl> impl;

		public:
		Variable(mlir::Value memory);

		bool operator==(const Variable& rhs) const;
		bool operator!=(const Variable& rhs) const;

		bool operator<(const Variable& rhs) const;
		bool operator>(const Variable& rhs) const;
		bool operator<=(const Variable& rhs) const;
		bool operator>=(const Variable& rhs) const;

		mlir::Value getReference();

		[[nodiscard]] bool isState() const;
		[[nodiscard]] bool isConstant() const;

		mlir::Value getDer();
		void setDer(mlir::Value value);

		[[nodiscard]] IndexSet toIndexSet() const;
		[[nodiscard]] MultiDimInterval toMultiDimInterval() const;

		[[nodiscard]] size_t indexOfElement(llvm::ArrayRef<size_t> access) const;
	};
}
