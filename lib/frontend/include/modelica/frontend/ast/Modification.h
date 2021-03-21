#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/SmallVector.h>
#include <variant>

#include "Expression.h"

namespace modelica
{
	class Argument;
	class ClassModification;
	class ElementModification;
	class ElementRedeclaration;
	class ElementReplaceable;
	class Modification;

	class Modification
	{
		public:
		explicit Modification(ClassModification classModification);
		Modification(ClassModification classModification, Expression expression);
		explicit Modification(Expression expression);

		[[nodiscard]] bool hasClassModification() const;
		[[nodiscard]] ClassModification& getClassModification();
		[[nodiscard]] const ClassModification& getClassModification() const;

		[[nodiscard]] bool hasExpression() const;
		[[nodiscard]] Expression& getExpression();
		[[nodiscard]] const Expression& getExpression() const;

		private:
		llvm::Optional<std::shared_ptr<ClassModification>> classModification;
		llvm::Optional<std::shared_ptr<Expression>> expression;
	};

	class ClassModification
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;
		template<typename T> using iterator = boost::indirect_iterator<typename Container<T>::iterator>;
		template<typename T>  using const_iterator = boost::indirect_iterator<typename Container<T>::const_iterator>;

		public:
		ClassModification(llvm::ArrayRef<Argument> arguments = {});

		[[nodiscard]] iterator<Argument> begin();
		[[nodiscard]] const_iterator<Argument> begin() const;

		[[nodiscard]] iterator<Argument> end();
		[[nodiscard]] const_iterator<Argument> end() const;

		private:
		Container<Argument> arguments;
	};

	class Argument
	{
		public:
		explicit Argument(ElementModification content);
		explicit Argument(ElementRedeclaration content);
		explicit Argument(ElementReplaceable content);

		template<typename T>
		[[nodiscard]] bool isA() const
		{
			return std::holds_alternative<std::shared_ptr<T>>(content);
		}

		template<typename T>
		[[nodiscard]] T& get()
		{
			assert(isA<T>());
			return *std::get<std::shared_ptr<T>>(content);
		}

		template<typename T>
		[[nodiscard]] const T& get() const
		{
			assert(isA<T>());
			return *std::get<std::shared_ptr<T>>(content);
		}

		private:
		std::variant<
		    std::shared_ptr<ElementModification>,
				std::shared_ptr<ElementRedeclaration>,
				std::shared_ptr<ElementReplaceable>> content;
	};

	class ElementModification
	{
		public:
		ElementModification(bool each, bool final, std::string name, Modification modification);
		ElementModification(bool each, bool final, std::string name);

		[[nodiscard]] bool hasEachProperty() const;
		[[nodiscard]] bool hasFinalProperty() const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] bool hasModification() const;
		[[nodiscard]] Modification& getModification();
		[[nodiscard]] const Modification& getModification() const;

		private:
		bool each;
		bool final;
		std::string name;
		llvm::Optional<std::shared_ptr<Modification>> modification;
	};

	// TODO: ElementReplaceable
	class ElementReplaceable
	{
	};

	// TODO: ElementRedeclaration
	class ElementRedeclaration
	{
	};
}
