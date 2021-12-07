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

		[[nodiscard]] mlir::Value getReference();
		[[nodiscard]] mlir::Value getState();
		[[nodiscard]] mlir::Value getDerivative();

		[[nodiscard]] int64_t getIdaOffset();
		[[nodiscard]] mlir::Value getIdaIndex();

		void setDerivative(Variable variable);
		void setTrivial(bool value);

		void setIdaOffset(int64_t offset);
		void setIdaIndex(mlir::Value index);

		[[nodiscard]] bool isState() const;
		[[nodiscard]] bool isConstant() const;
		[[nodiscard]] bool isDerivative() const;
		[[nodiscard]] bool isTrivial() const;

		[[nodiscard]] bool isTime() const;

		[[nodiscard]] bool hasIdaOffset() const;
		[[nodiscard]] bool hasIdaIndex() const;

		[[nodiscard]] IndexSet toIndexSet() const;
		[[nodiscard]] MultiDimInterval toMultiDimInterval() const;

		[[nodiscard]] size_t indexOfElement(llvm::ArrayRef<size_t> access) const;
	};
}
