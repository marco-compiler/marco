#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

namespace modelica
{
	class Expression;

	class Tuple
	{
		public:
		using UniqueExpr = std::unique_ptr<Expression>;

		explicit Tuple(std::initializer_list<Expression> expressions);

		template<typename Iter>
		Tuple(Iter begin, Iter end)
		{
			for (auto it = begin; it != end; it++)
				expressions.push_back(std::make_unique<Expression>(*it));
		}

		Tuple(const Tuple& other);
		Tuple(Tuple&& other) = default;

		~Tuple() = default;

		Tuple& operator=(const Tuple& other);
		Tuple& operator=(Tuple&& other) = default;

		[[nodiscard]] bool operator==(const Tuple& other) const;
		[[nodiscard]] bool operator!=(const Tuple& other) const;

		[[nodiscard]] UniqueExpr& operator[](size_t index);
		[[nodiscard]] const UniqueExpr& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] int size() const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::iterator begin();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::const_iterator begin()
				const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::iterator end();
		[[nodiscard]] llvm::SmallVectorImpl<UniqueExpr>::const_iterator end() const;

		private:
		llvm::SmallVector<UniqueExpr, 3> expressions;
	};
}	 // namespace modelica
