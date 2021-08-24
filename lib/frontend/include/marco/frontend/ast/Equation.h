#pragma once

#include <memory>

#include "ASTNode.h"

namespace marco::frontend
{
	class Expression;

	class Equation
			: public ASTNode,
				public impl::Cloneable<Equation>,
				public impl::Dumpable<Equation>
	{
		public:
		template<typename... Args>
		static std::unique_ptr<Equation> build(Args&&... args)
		{
			return std::unique_ptr<Equation>(new Equation(std::forward<Args>(args)...));
		}

		Equation(const Equation& other);
		Equation(Equation&& other);
		~Equation() override;

		Equation& operator=(const Equation& other);
		Equation& operator=(Equation&& other);

		friend void swap(Equation& first, Equation& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Expression* getLhsExpression();
		[[nodiscard]] const Expression* getLhsExpression() const;
		void setLhsExpression(std::unique_ptr<Expression> expression);

		[[nodiscard]] Expression* getRhsExpression();
		[[nodiscard]] const Expression* getRhsExpression() const;
		void setRhsExpression(std::unique_ptr<Expression> expression);

		private:
		Equation(SourceRange location,
						 std::unique_ptr<Expression> lhs,
						 std::unique_ptr<Expression> rhs);

		std::unique_ptr<Expression> lhs;
		std::unique_ptr<Expression> rhs;
	};
}
