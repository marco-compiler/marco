#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>

#include "ASTNode.h"

namespace modelica::frontend
{
	class Statement;

	class Algorithm
			: public impl::ASTNodeCRTP<Algorithm>,
				public impl::Cloneable<Algorithm>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using statements_iterator = Container<std::unique_ptr<Statement>>::iterator;
		using statements_const_iterator = Container<std::unique_ptr<Statement>>::const_iterator;

		Algorithm(SourcePosition location,
							llvm::ArrayRef<std::unique_ptr<Statement>> statements);

		Algorithm(const Algorithm& other);
		Algorithm(Algorithm&& other);

		~Algorithm() override;

		Algorithm& operator=(const Algorithm& other);
		Algorithm& operator=(Algorithm&& other);

		friend void swap(Algorithm& first, Algorithm& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::ALGORITHM;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		Statement* operator[](size_t index);
		const Statement* operator[](size_t index) const;

		[[nodiscard]] llvm::StringRef getReturnCheckName() const;
		void setReturnCheckName(llvm::StringRef name);

		[[nodiscard]] size_t size() const;

		[[nodiscard]] statements_iterator begin();
		[[nodiscard]] statements_const_iterator begin() const;

		[[nodiscard]] statements_iterator end();
		[[nodiscard]] statements_const_iterator end() const;

		private:
		std::string returnCheckName;
		Container<std::unique_ptr<Statement>> statements;
	};
}
