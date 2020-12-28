#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/frontend/Statement.hpp>
#include <modelica/utils/SourceRange.hpp>

namespace modelica
{
	class Algorithm
	{
		private:
		using Container = llvm::SmallVector<Statement, 3>;

		public:
		using statements_iterator = Container::iterator;
		using statements_const_iterator = Container::const_iterator;

		explicit Algorithm(llvm::ArrayRef<Statement> statements);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] Container& getStatements();
		[[nodiscard]] const Container& getStatements() const;

		[[nodiscard]] statements_iterator begin();
		[[nodiscard]] statements_const_iterator begin() const;

		[[nodiscard]] statements_iterator end();
		[[nodiscard]] statements_const_iterator end() const;

		private:
		Container statements;
	};
}	 // namespace modelica
