#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>
#include <memory>

#include "Expression.h"

namespace modelica::frontend
{
	/**
	 * A tuple is a container for destinations of a call. It is NOT an
	 * array-like structure that is supposed to be summable, passed around or
	 * whatever.
	 */
	class Tuple
			: public impl::ExpressionCRTP<Tuple>,
				public impl::Cloneable<Tuple>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<std::unique_ptr<Expression>>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<std::unique_ptr<Expression>>::const_iterator>;

		Tuple(SourcePosition location,
					llvm::ArrayRef<std::unique_ptr<Expression>> expressions = llvm::None);

		Tuple(const Tuple& other);
		Tuple(Tuple&& other);
		~Tuple() override;

		Tuple& operator=(const Tuple& other);
		Tuple& operator=(Tuple&& other);

		friend void swap(Tuple& first, Tuple& second);

		/*
		template<typename Iter>
		Tuple(SourcePosition location, Iter begin, Iter end)
				: location(std::move(location))
		{
			for (auto it = begin; it != end; ++it)
				expressions.push_back(std::make_shared<Expression>(*it));
		}
		 */

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::EXPRESSION_TUPLE;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool isLValue() const override;

		[[nodiscard]] bool operator==(const Tuple& other) const;
		[[nodiscard]] bool operator!=(const Tuple& other) const;

		[[nodiscard]] Expression* operator[](size_t index);
		[[nodiscard]] const Expression* operator[](size_t index) const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container<std::unique_ptr<Expression>> expressions;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Tuple& obj);

	std::string toString(const Tuple& obj);
}
