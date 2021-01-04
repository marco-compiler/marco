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
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using statements_iterator = Container<Statement>::iterator;
		using statements_const_iterator = Container<Statement>::const_iterator;

		Algorithm(SourcePosition location, llvm::ArrayRef<Statement> statements);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] Container<Statement>& getStatements();
		[[nodiscard]] const Container<Statement>& getStatements() const;

		[[nodiscard]] statements_iterator begin();
		[[nodiscard]] statements_const_iterator begin() const;

		[[nodiscard]] statements_iterator end();
		[[nodiscard]] statements_const_iterator end() const;

		private:
		SourcePosition location;
		Container<Statement> statements;
	};
}	 // namespace modelica
