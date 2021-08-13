#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <memory>
#include <mlir/IR/Value.h>
#include <marco/utils/IndexSet.hpp>

namespace marco::codegen::model
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
		[[nodiscard]] bool isDerivative() const;
		[[nodiscard]] bool isTrivial() const;

		[[nodiscard]] bool isTime() const;

		[[nodiscard]] mlir::Value getState();
		[[nodiscard]] mlir::Value getDer();

		void setDer(Variable variable);
		void setTrivial(bool value);

		[[nodiscard]] IndexSet toIndexSet() const;
		[[nodiscard]] MultiDimInterval toMultiDimInterval() const;

		[[nodiscard]] size_t indexOfElement(llvm::ArrayRef<size_t> access) const;
	};
}
