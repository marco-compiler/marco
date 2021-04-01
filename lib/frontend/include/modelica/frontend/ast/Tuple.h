#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>
#include <memory>

namespace modelica::frontend
{
	class Expression;

	/**
	 * A tuple is a container for destinations of a call. It is NOT an
	 * array-like structure that is supposed to be summable, passed around or
	 * whatever.
	 */
	class Tuple
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		using iterator = boost::indirect_iterator<Container<Expression>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<Expression>::const_iterator>;

		explicit Tuple(SourcePosition location, llvm::ArrayRef<Expression> expressions = {});

		template<typename Iter>
		Tuple(SourcePosition location, Iter begin, Iter end)
				: location(std::move(location))
		{
			for (auto it = begin; it != end; ++it)
				expressions.push_back(std::make_shared<Expression>(*it));
		}

		Tuple(const Tuple& other);
		Tuple(Tuple&& other) = default;

		~Tuple() = default;

		Tuple& operator=(const Tuple& other);
		Tuple& operator=(Tuple&& other) = default;

		[[nodiscard]] bool operator==(const Tuple& other) const;
		[[nodiscard]] bool operator!=(const Tuple& other) const;

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
		Container<Expression> expressions;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Tuple& obj);

	std::string toString(const Tuple& obj);
}
