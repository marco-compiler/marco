#pragma once

#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>

namespace modelica::frontend
{

	/**
	 * Enum for LLVM-style RTTI.
	 *
	 * Note that the values have a specific order, and abstract classes rely
	 * on that to determine whether the object is one of its subclasses.
	 */
	enum class ASTNodeKind
	{
		ALGORITHM,
		ANNOTATION,
		CLASS,
		FUNCTION,
		FUNCTION_DER,
		FUNCTION_STANDARD,
		FUNCTION_LAST,
		CLASS_MODEL,
		CLASS_PACKAGE,
		CLASS_RECORD,
		CLASS_LAST,
		EQUATION,
		EXPRESSION,
		EXPRESSION_ARRAY,
		EXPRESSION_CALL,
		EXPRESSION_CONSTANT,
		EXPRESSION_OPERATION,
		EXPRESSION_REFERENCE_ACCESS,
		EXPRESSION_TUPLE,
		EXPRESSION_LAST_EXPRESSION,
		FOR_EQUATION,
		INDUCTION,
		MEMBER,
		MODIFICATION,
		CLASS_MODIFICATION,
		ARGUMENT,
		ARGUMENT_ELEMENT_MODIFICATION,
		ARGUMENT_ELEMENT_REPLACEABLE,
		ARGUMENT_ELEMENT_REDECLARATION,
		ARGUMENT_LAST,
		STATEMENT,
		STATEMENT_ASSIGNMENT,
		STATEMENT_BREAK,
		STATEMENT_FOR,
		STATEMENT_IF,
		STATEMENT_WHEN,
		STATEMENT_WHILE,
		STATEMENT_RETURN,
		STATEMENT_LAST
	};

	class ASTNode;

	namespace impl
	{
		class ASTNodeBase
		{
			public:
			friend class ::modelica::frontend::ASTNode;

			virtual void dump(llvm::raw_ostream& os, size_t indents = 0) const = 0;

			[[nodiscard]] SourcePosition getLocation() const;

			protected:
			ASTNodeBase(const ASTNodeBase& other);
			ASTNodeBase(ASTNodeBase&& other);

			ASTNodeBase& operator=(const ASTNodeBase& other);
			ASTNodeBase& operator=(ASTNodeBase&& other);

			private:
			void setLocation(SourcePosition location);

			ASTNodeKind kind;
			SourcePosition location;
		};
	}

	class ASTNode
	{
		public:
		ASTNode(ASTNodeKind kind, SourcePosition location);

		virtual ~ASTNode();

		friend void swap(ASTNode& first, ASTNode& second);

		[[nodiscard]] ASTNodeKind getKind() const
		{
			return node->kind;
		}

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return llvm::isa<T>(*node);
		}

		template<typename T>
		[[nodiscard]] T* dyn_cast()
		{
			return llvm::dyn_cast<T>(node.get());
		}

		template<typename T>
		[[nodiscard]] const T* dyn_cast() const
		{
			return llvm::dyn_cast<T>(node.get());
		}

		template<typename T>
		[[nodiscard]] T* cast()
		{
			return llvm::cast<T>(node.get());
		}

		template<typename T>
		[[nodiscard]] const T* cast() const
		{
			return llvm::cast<T>(node.get());
		}

		void dump() const;

		virtual void dump(llvm::raw_ostream& os, size_t indents = 0) const = 0;

		[[nodiscard]] SourcePosition getLocation() const;

		protected:
		ASTNode(const ASTNode& other);
		ASTNode(ASTNode&& other);

		ASTNode& operator=(const ASTNode& other);
		ASTNode& operator=(ASTNode&& other);

		private:
		std::unique_ptr<impl::ASTNodeBase> node;
	};

	namespace impl
	{
		template<class Derived>
		struct ASTNodeCRTP : public ASTNodeBase
		{
			public:
			using ASTNodeBase::ASTNodeBase;

			[[nodiscard]] std::unique_ptr<ASTNode> cloneNode() const
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};

		template<class Derived>
		struct Cloneable
		{
			[[nodiscard]] std::unique_ptr<Derived> clone() const
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
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
}