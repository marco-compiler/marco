#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "ASTNode.h"

namespace modelica::frontend
{
	class Equation;
	class Induction;

	/**
	 * "For equations" are different with respect to regular equations
	 * because they introduce a set of inductions, and thus a new set of names
	 * available within the for cycle.
	 *
	 * Inductions are mapped to a set of indexes so that an from a name we can
	 * deduce a index and from a index we can deduce a name
	 */
	class ForEquation
			: public impl::ASTNodeCRTP<ForEquation>,
				public impl::Cloneable<ForEquation>
	{
		public:
		ForEquation(SourcePosition location,
								llvm::ArrayRef<std::unique_ptr<Induction>> inductions,
								std::unique_ptr<Equation> equation);

		ForEquation(const ForEquation& other);
		ForEquation(ForEquation&& other);
		~ForEquation() override;

		ForEquation& operator=(const ForEquation& other);
		ForEquation& operator=(ForEquation&& other);

		friend void swap(ForEquation& first, ForEquation& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::FOR_EQUATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Induction>> getInductions();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Induction>> getInductions() const;
		[[nodiscard]] size_t inductionsCount() const;

		[[nodiscard]] Equation* getEquation() const;

		private:
		llvm::SmallVector<std::unique_ptr<Induction>, 3> inductions;
		std::unique_ptr<Equation> equation;
	};
}
