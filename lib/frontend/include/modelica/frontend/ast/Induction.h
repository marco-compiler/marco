#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

#include "Equation.h"
#include "Expression.h"

namespace modelica::frontend
{
	/**
	 * An induction is the in memory of a piece of code such as
	 * for i in 0:20. and induction holds a name and the begin and end
	 * expressions.
	 *
	 * Notice that for the compiler we made the assumption that all range will be
	 * of step one.
	 */
	class Induction
			: public impl::ASTNodeCRTP<Induction>,
				public impl::Cloneable<Induction>
	{
		public:
		Induction(SourcePosition location,
							llvm::StringRef inductionVariable,
							std::unique_ptr<Expression> begin,
							std::unique_ptr<Expression> end);

		Induction(const Induction& other);
		Induction(Induction&& other);
		~Induction() override;

		Induction& operator=(const Induction& other);
		Induction& operator=(Induction&& other);

		friend void swap(Induction& first, Induction& second);

		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] Expression* getBegin();
		[[nodiscard]] const Expression* getBegin() const;

		[[nodiscard]] Expression* getEnd();
		[[nodiscard]] const Expression* getEnd() const;

		[[nodiscard]] size_t getInductionIndex() const;
		void setInductionIndex(size_t index);

		private:
		std::string inductionVariable;
		std::unique_ptr<Expression> begin;
		std::unique_ptr<Expression> end;
		size_t inductionIndex;
	};
}
