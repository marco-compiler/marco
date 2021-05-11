#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <variant>

#include "Expression.h"

namespace modelica::frontend
{
	class Argument;
	class ClassModification;
	class ElementModification;
	class ElementRedeclaration;
	class ElementReplaceable;
	class Modification;

	class Modification
			: public impl::ASTNodeCRTP<Modification>,
				public impl::Cloneable<Modification>
	{
		public:
		Modification(SourcePosition location,
								 std::unique_ptr<ClassModification> classModification);

		Modification(SourcePosition location,
								 std::unique_ptr<ClassModification> classModification,
								 std::unique_ptr<Expression> expression);

		Modification(SourcePosition location,
								 std::unique_ptr<Expression> expression);

		Modification(const Modification& other);
		Modification(Modification&& other);
		~Modification() override;

		Modification& operator=(const Modification& other);
		Modification& operator=(Modification&& other);

		friend void swap(Modification& first, Modification& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::MODIFICATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool hasClassModification() const;
		[[nodiscard]] ClassModification* getClassModification();
		[[nodiscard]] const ClassModification* getClassModification() const;

		[[nodiscard]] bool hasExpression() const;
		[[nodiscard]] Expression* getExpression();
		[[nodiscard]] const Expression* getExpression() const;

		private:
		llvm::Optional<std::unique_ptr<ClassModification>> classModification;
		llvm::Optional<std::unique_ptr<Expression>> expression;
	};

	class ClassModification
			: public impl::ASTNodeCRTP<ClassModification>,
				public impl::Cloneable<ClassModification>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Argument>>::iterator;
		using const_iterator = Container<std::unique_ptr<Argument>>::const_iterator;

		ClassModification(SourcePosition location,
											llvm::ArrayRef<std::unique_ptr<Argument>> arguments = llvm::None);

		ClassModification(const ClassModification& other);
		ClassModification(ClassModification&& other);
		~ClassModification() override;

		ClassModification& operator=(const ClassModification& other);
		ClassModification& operator=(ClassModification&& other);

		friend void swap(ClassModification& first, ClassModification& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::CLASS_MODIFICATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		Container<std::unique_ptr<Argument>> arguments;
	};

	class Argument : public impl::ASTNodeCRTP<Argument>
	{
		public:
		Argument(ASTNodeKind kind, SourcePosition location);

		Argument(const Argument& other);
		Argument(Argument&& other);
		~Argument() override;

		Argument& operator=(const Argument& other);
		Argument& operator=(Argument&& other);

		friend void swap(Argument& first, Argument& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::ARGUMENT &&
					   node->getKind() <= ASTNodeKind::ARGUMENT_LAST;
		}

		[[nodiscard]] virtual std::unique_ptr<Argument> cloneArgument() const = 0;
	};

	namespace impl
	{
		template<class Derived>
		struct ArgumentCRTP : public Argument
		{
			public:
			using Argument::Argument;

			[[nodiscard]] std::unique_ptr<Argument> cloneArgument() const override
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};
	}

	class ElementModification
			: public impl::ArgumentCRTP<ElementModification>,
				public impl::Cloneable<ElementModification>
	{
		public:
		ElementModification(SourcePosition location,
												bool each,
												bool final,
												llvm::StringRef name,
												std::unique_ptr<Modification>& modification);

		ElementModification(SourcePosition location,
												bool each,
												bool final,
												llvm::StringRef name);

		ElementModification(const ElementModification& other);
		ElementModification(ElementModification&& other);
		~ElementModification() override;

		ElementModification& operator=(const ElementModification& other);
		ElementModification& operator=(ElementModification&& other);

		friend void swap(ElementModification& first, ElementModification& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::ARGUMENT_ELEMENT_MODIFICATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool hasEachProperty() const;
		[[nodiscard]] bool hasFinalProperty() const;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] bool hasModification() const;
		[[nodiscard]] Modification* getModification();
		[[nodiscard]] const Modification* getModification() const;

		private:
		bool each;
		bool final;
		std::string name;
		llvm::Optional<std::unique_ptr<Modification>> modification;
	};

	// TODO: ElementReplaceable
	class ElementReplaceable
			: public impl::ArgumentCRTP<ElementReplaceable>,
				public impl::Cloneable<ElementReplaceable>
	{
		public:
		explicit ElementReplaceable(SourcePosition location);
		ElementReplaceable(const ElementReplaceable& other);
		ElementReplaceable(ElementReplaceable&& other);
		~ElementReplaceable() override;

		ElementReplaceable& operator=(const ElementReplaceable& other);
		ElementReplaceable& operator=(ElementReplaceable&& other);

		friend void swap(ElementReplaceable& first, ElementReplaceable& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::ARGUMENT_ELEMENT_REPLACEABLE;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;
	};

	// TODO: ElementRedeclaration
	class ElementRedeclaration
			: public impl::ArgumentCRTP<ElementRedeclaration>,
				public impl::Cloneable<ElementRedeclaration>
	{
		public:
		explicit ElementRedeclaration(SourcePosition location);
		ElementRedeclaration(const ElementRedeclaration& other);
		ElementRedeclaration(ElementRedeclaration&& other);
		~ElementRedeclaration() override;

		ElementRedeclaration& operator=(const ElementRedeclaration& other);
		ElementRedeclaration& operator=(ElementRedeclaration&& other);

		friend void swap(ElementRedeclaration& first, ElementRedeclaration& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() == ASTNodeKind::ARGUMENT_ELEMENT_REDECLARATION;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;
	};
}
