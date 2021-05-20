#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <variant>

#include "ASTNode.h"

namespace modelica::frontend
{
	class Argument;
	class ClassModification;
	class ElementModification;
	class ElementRedeclaration;
	class ElementReplaceable;
	class Modification;

	class Modification
			: public ASTNode,
				public impl::Cloneable<Modification>,
				public impl::Dumpable<Modification>
	{
		public:
		template<typename... Args>
		static std::unique_ptr<Modification> build(Args&&... args)
		{
			return std::unique_ptr<Modification>(new Modification(std::forward<Args>(args)...));
		}

		Modification(const Modification& other);
		Modification(Modification&& other);
		~Modification() override;

		Modification& operator=(const Modification& other);
		Modification& operator=(Modification&& other);

		friend void swap(Modification& first, Modification& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool hasClassModification() const;
		[[nodiscard]] ClassModification* getClassModification();
		[[nodiscard]] const ClassModification* getClassModification() const;

		[[nodiscard]] bool hasExpression() const;
		[[nodiscard]] Expression* getExpression();
		[[nodiscard]] const Expression* getExpression() const;

		private:
		Modification(SourceRange location,
								 std::unique_ptr<ClassModification> classModification);

		Modification(SourceRange location,
								 std::unique_ptr<ClassModification> classModification,
								 std::unique_ptr<Expression> expression);

		Modification(SourceRange location,
								 std::unique_ptr<Expression> expression);

		llvm::Optional<std::unique_ptr<ClassModification>> classModification;
		llvm::Optional<std::unique_ptr<Expression>> expression;
	};

	class ClassModification
			: public ASTNode,
				public impl::Cloneable<ClassModification>,
				public impl::Dumpable<ClassModification>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Argument>>::iterator;
		using const_iterator = Container<std::unique_ptr<Argument>>::const_iterator;

		template<typename... Args>
		static std::unique_ptr<ClassModification> build(Args&&... args)
		{
			return std::unique_ptr<ClassModification>(new ClassModification(std::forward<Args>(args)...));
		}

		ClassModification(const ClassModification& other);
		ClassModification(ClassModification&& other);
		~ClassModification() override;

		ClassModification& operator=(const ClassModification& other);
		ClassModification& operator=(ClassModification&& other);

		friend void swap(ClassModification& first, ClassModification& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		ClassModification(SourceRange location,
											llvm::ArrayRef<std::unique_ptr<Argument>> arguments = llvm::None);

		Container<std::unique_ptr<Argument>> arguments;
	};

	class ElementModification
			: public ASTNode,
				public impl::Dumpable<ElementModification>
	{
		public:
		ElementModification(const ElementModification& other);
		ElementModification(ElementModification&& other);
		~ElementModification() override;

		ElementModification& operator=(const ElementModification& other);
		ElementModification& operator=(ElementModification&& other);

		friend void swap(ElementModification& first, ElementModification& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool hasEachProperty() const;
		[[nodiscard]] bool hasFinalProperty() const;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] bool hasModification() const;
		[[nodiscard]] Modification* getModification();
		[[nodiscard]] const Modification* getModification() const;

		private:
		friend class Argument;

		ElementModification(SourceRange location,
												bool each,
												bool final,
												llvm::StringRef name,
												std::unique_ptr<Modification> modification);

		ElementModification(SourceRange location,
												bool each,
												bool final,
												llvm::StringRef name);

		bool each;
		bool final;
		std::string name;
		llvm::Optional<std::unique_ptr<Modification>> modification;
	};

	// TODO: ElementReplaceable
	class ElementReplaceable
			: public ASTNode,
				public impl::Dumpable<ElementReplaceable>
	{
		public:
		ElementReplaceable(const ElementReplaceable& other);
		ElementReplaceable(ElementReplaceable&& other);
		~ElementReplaceable() override;

		ElementReplaceable& operator=(const ElementReplaceable& other);
		ElementReplaceable& operator=(ElementReplaceable&& other);

		friend void swap(ElementReplaceable& first, ElementReplaceable& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		private:
		friend class Argument;

		explicit ElementReplaceable(SourceRange location);
	};

	// TODO: ElementRedeclaration
	class ElementRedeclaration
			: public ASTNode,
				public impl::Dumpable<ElementRedeclaration>
	{
		public:
		ElementRedeclaration(const ElementRedeclaration& other);
		ElementRedeclaration(ElementRedeclaration&& other);
		~ElementRedeclaration() override;

		ElementRedeclaration& operator=(const ElementRedeclaration& other);
		ElementRedeclaration& operator=(ElementRedeclaration&& other);

		friend void swap(ElementRedeclaration& first, ElementRedeclaration& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		private:
		friend class Argument;

		explicit ElementRedeclaration(SourceRange location);
	};

	class Argument
			: public impl::Cloneable<Argument>,
				public impl::Dumpable<Argument>
	{
		public:
		template<typename... Args>
		static std::unique_ptr<Algorithm> build(Args&&... args)
		{
			return std::make_unique<Algorithm>(std::forward<Args>(args)...);
		}

		Argument(const Argument& other);
		Argument(Argument&& other);
		~Argument();

		Argument& operator=(const Argument& other);
		Argument& operator=(Argument&& other);

		friend void swap(Argument& first, Argument& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		template<typename T>
		[[nodiscard]] bool isa() const
		{
			return std::holds_alternative<T>(content);
		}

		template<typename T>
		[[nodiscard]] T* get()
		{
			return &std::get<T>(content);
		}

		template<typename T>
		[[nodiscard]] const T* get() const
		{
			return &std::get<T>(content);
		}

		template<typename Visitor>
		void visit(Visitor&& visitor)
		{
			std::visit(visitor, content);
		}

		template<typename Visitor>
		void visit(Visitor&& visitor) const
		{
			std::visit(visitor, content);
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Argument> elementModification(Args&&... args)
		{
			ElementModification content(std::forward<Args>(args)...);
			return std::unique_ptr<Argument>(new Argument(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Argument> elementRedeclaration(Args&&... args)
		{
			ElementRedeclaration content(std::forward<Args>(args)...);
			return std::unique_ptr<Argument>(new Argument(std::move(content)));
		}

		template<typename... Args>
		[[nodiscard]] static std::unique_ptr<Argument> elementReplaceable(Args&&... args)
		{
			ElementReplaceable content(std::forward<Args>(args)...);
			return std::unique_ptr<Argument>(new Argument(std::move(content)));
		}

		private:
		explicit Argument(ElementModification content);
		explicit Argument(ElementRedeclaration content);
		explicit Argument(ElementReplaceable content);

		std::variant<
		    ElementModification,
				ElementRedeclaration,
				ElementReplaceable> content;
	};
}
