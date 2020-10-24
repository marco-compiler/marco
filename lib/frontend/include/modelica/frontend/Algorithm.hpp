#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <vector>

#include "Statement.hpp"

namespace modelica
{
	class Algorithm
	{
		public:
		Algorithm(llvm::ArrayRef<Statement> statements);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] const llvm::SmallVectorImpl<Statement>& getStatements() const;

		private:
		llvm::SmallVector<Statement, 3> statements;
	};
}	 // namespace modelica
