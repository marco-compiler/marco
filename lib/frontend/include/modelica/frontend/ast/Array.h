#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <modelica/utils/SourcePosition.h>
#include <memory>

namespace modelica::frontend
{
	class Expression;

	class Array
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<Expression>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<Expression>::const_iterator>;

		Array(SourcePosition location, llvm::ArrayRef<Expression> values);

		[[nodiscard]] bool operator==(const Array& other) const;
		[[nodiscard]] bool operator!=(const Array& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin()const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		SourcePosition location;
		Container<Expression> values;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Array& obj);

	std::string toString(const Array& obj);
}
