#ifndef MARCO_AST_NODE_FOREQUATION_H
#define MARCO_AST_NODE_FOREQUATION_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

namespace marco::ast
{
	class Equation;
	class Induction;

	/// "For equations" are different with respect to regular equations
	/// because they introduce a set of inductions, and thus a new set of names
	/// available within the for cycle.
	///
	/// Inductions are mapped to a set of indexes so that an from a name we can
	/// deduce a index and from a index we can deduce a name.
	class ForEquation
			: public ASTNode,
				public impl::Cloneable<ForEquation>,
				public impl::Dumpable<ForEquation>
	{
		public:
      template<typename... Args>
      static std::unique_ptr<ForEquation> build(Args&&... args)
      {
        return std::unique_ptr<ForEquation>(new ForEquation(std::forward<Args>(args)...));
      }

      ForEquation(const ForEquation& other);
      ForEquation(ForEquation&& other);
      ~ForEquation() override;

      ForEquation& operator=(const ForEquation& other);
      ForEquation& operator=(ForEquation&& other);

      friend void swap(ForEquation& first, ForEquation& second);

      void print(llvm::raw_ostream& os, size_t indents = 0) const override;

      [[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Induction>> getInductions();
      [[nodiscard]] llvm::ArrayRef<std::unique_ptr<Induction>> getInductions() const;
      [[nodiscard]] size_t inductionsCount() const;

      void addOuterInduction(std::unique_ptr<Induction> induction);

      [[nodiscard]] Equation* getEquation() const;

		private:
      ForEquation(SourceRange location,
                  llvm::ArrayRef<std::unique_ptr<Induction>> inductions,
                  std::unique_ptr<Equation> equation);

    private:
      llvm::SmallVector<std::unique_ptr<Induction>, 3> inductions;
      std::unique_ptr<Equation> equation;
	};
}

#endif // MARCO_AST_NODE_FOREQUATION_H
