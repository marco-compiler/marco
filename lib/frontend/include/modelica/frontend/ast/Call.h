#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourceRange.hpp>

namespace modelica
{
	class Expression;

	class Call
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		using args_iterator = boost::indirect_iterator<Container<Expression>::iterator>;
		using args_const_iterator = boost::indirect_iterator<Container<Expression>::const_iterator>;

		Call(SourcePosition location, Expression function, llvm::ArrayRef<Expression> args = {}, unsigned int elementWiseRank = 0);

		[[nodiscard]] bool operator==(const Call& other) const;
		[[nodiscard]] bool operator!=(const Call& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] Expression& getFunction();
		[[nodiscard]] const Expression& getFunction() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] args_iterator begin();
		[[nodiscard]] args_const_iterator begin() const;

		[[nodiscard]] args_iterator end();
		[[nodiscard]] args_const_iterator end() const;

		[[nodiscard]] bool isElementWise() const;
		[[nodiscard]] unsigned int getElementWiseRank() const;
		void setElementWiseRank(unsigned int rank);

		private:
		SourcePosition location;
		std::shared_ptr<Expression> function;
		Container<Expression> args;
		unsigned int elementWiseRank;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Call& obj);

	std::string toString(const Call& obj);
}	 // namespace modelica
