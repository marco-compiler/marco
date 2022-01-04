#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "ASTNode.h"
#include "Type.h"

namespace marco::ast
{
	class Expression;

	/**
	 * A tuple is a container for destinations of a call. It is NOT an
	 * array-like structure that is supposed to be summable, passed around or
	 * whatever.
	 */
	class Tuple
			: public ASTNode,
				public impl::Dumpable<Tuple>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Expression>>::iterator;
		using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		Tuple(const Tuple& other);
		Tuple(Tuple&& other);
		~Tuple() override;

		Tuple& operator=(const Tuple& other);
		Tuple& operator=(Tuple&& other);

		friend void swap(Tuple& first, Tuple& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] bool operator==(const Tuple& other) const;
		[[nodiscard]] bool operator!=(const Tuple& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] Expression* getArg(size_t index);
		[[nodiscard]] const Expression* getArg(size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		friend class Expression;

		Tuple(SourceRange location,
					Type type,
					llvm::ArrayRef<std::unique_ptr<Expression>> expressions = llvm::None);

		Type type;
		Container<std::unique_ptr<Expression>> expressions;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Tuple& obj);

	std::string toString(const Tuple& obj);
}
