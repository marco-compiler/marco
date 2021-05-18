#pragma once

#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>

namespace modelica::frontend
{
	namespace impl
	{
		template<class Derived>
		struct Cloneable
		{
			[[nodiscard]] std::unique_ptr<Derived> clone() const
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};

		template<class Derived>
		struct Dumpable
		{
			void dump() const
			{
				dump(llvm::outs(), 0);
			}

			void dump(llvm::raw_ostream& os, size_t indents = 0) const
			{
				print(os, indents);
			}

			virtual void print(llvm::raw_ostream& os, size_t indents = 0) const = 0;
		};

		template<typename T>
		void swap(llvm::SmallVectorImpl<std::unique_ptr<T>>& first,
							llvm::SmallVectorImpl<std::unique_ptr<T>>& second)
		{
			llvm::SmallVector<std::unique_ptr<T>, 3> tmp;

			tmp = std::move(first);
			first = std::move(second);
			second = std::move(tmp);
		}
	}

	class ASTNode
	{
		public:
		ASTNode(SourcePosition location);

		ASTNode(const ASTNode& other);
		ASTNode(ASTNode&& other);

		virtual ~ASTNode() = 0;

		ASTNode& operator=(const ASTNode& other);
		ASTNode& operator=(ASTNode&& other);

		friend void swap(ASTNode& first, ASTNode& second);

		[[nodiscard]] SourcePosition getLocation() const;

		private:
		SourcePosition location;
	};
}
