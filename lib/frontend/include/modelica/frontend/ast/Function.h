#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <memory>
#include <string>

#include "ASTNode.h"
#include "Class.h"

namespace modelica::frontend
{
	class Algorithm;
	class Annotation;
	class Member;

	class Function : public ASTNode
	{
		public:
		Function(SourceRange location, llvm::StringRef name);

		Function(const Function& other);
		Function(Function&& other);

		~Function() override;

		Function& operator=(const Function& other);
		Function& operator=(Function&& other);

		friend void swap(Function& first, Function& second);

		[[nodiscard]] llvm::StringRef getName() const;

		private:
		std::string name;
	};

	class PartialDerFunction
			: public Function,
				public impl::Dumpable<PartialDerFunction>
	{
		public:
		PartialDerFunction(const PartialDerFunction& other);
		PartialDerFunction(PartialDerFunction&& other);
		~PartialDerFunction() override;

		PartialDerFunction& operator=(const PartialDerFunction& other);
		PartialDerFunction& operator=(PartialDerFunction&& other);

		friend void swap(PartialDerFunction& first, PartialDerFunction& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Expression* getDerivedFunction() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Expression>> getIndependentVariables();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Expression>> getIndependentVariables() const;

		[[nodiscard]] llvm::MutableArrayRef<Type> getArgsTypes();
		[[nodiscard]] llvm::ArrayRef<Type> getArgsTypes() const;
		void setArgsTypes(llvm::ArrayRef<Type> types);

		[[nodiscard]] llvm::MutableArrayRef<Type> getResultsTypes();
		[[nodiscard]] llvm::ArrayRef<Type> getResultsTypes() const;
		void setResultsTypes(llvm::ArrayRef<Type> types);

		private:
		friend class Class;

		PartialDerFunction(SourceRange location,
											 llvm::StringRef name,
											 std::unique_ptr<Expression> derivedFunction,
											 llvm::ArrayRef<std::unique_ptr<Expression>> independentVariables);

		std::unique_ptr<Expression> derivedFunction;
		llvm::SmallVector<std::unique_ptr<Expression>, 3> independentVariables;
		llvm::SmallVector<Type, 3> args;
		llvm::SmallVector<Type, 3> results;
	};

	class StandardFunction
			: public Function,
				public impl::Dumpable<StandardFunction>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		StandardFunction(const StandardFunction& other);
		StandardFunction(StandardFunction&& other);
		~StandardFunction() override;

		StandardFunction& operator=(const StandardFunction& other);
		StandardFunction& operator=(StandardFunction&& other);

		friend void swap(StandardFunction& first, StandardFunction& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Member* operator[](llvm::StringRef name);
		[[nodiscard]] const Member* operator[](llvm::StringRef name) const;

		[[nodiscard]] bool isPure() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Member>> getMembers();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Member>> getMembers() const;

		[[nodiscard]] Container<Member*> getArgs() const;
		[[nodiscard]] Container<Member*> getResults() const;
		[[nodiscard]] Container<Member*> getProtectedMembers() const;

		void addMember(std::unique_ptr<Member> member);

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Algorithm>> getAlgorithms();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Algorithm>> getAlgorithms() const;

		[[nodiscard]] bool hasAnnotation() const;
		[[nodiscard]] Annotation* getAnnotation();
		[[nodiscard]] const Annotation* getAnnotation() const;

		[[nodiscard]] FunctionType getType() const;

		private:
		friend class Class;

		StandardFunction(SourceRange location,
										 bool pure,
										 llvm::StringRef name,
										 llvm::ArrayRef<std::unique_ptr<Member>> members,
										 llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms,
										 llvm::Optional<std::unique_ptr<Annotation>> annotation = llvm::None);

		bool pure;
		Container<std::unique_ptr<Member>> members;
		Container<std::unique_ptr<Algorithm>> algorithms;
		llvm::Optional<std::unique_ptr<Annotation>> annotation;
	};

	class DerivativeAnnotation
	{
		public:
		DerivativeAnnotation(llvm::StringRef name, unsigned int order = 1);

		[[nodiscard]] llvm::StringRef getName() const;
		[[nodiscard]] unsigned int getOrder() const;

		private:
		std::string name;
		unsigned int order;
	};

	class InverseFunctionAnnotation
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		InverseFunctionAnnotation();

		[[nodiscard]] bool isInvertible(llvm::StringRef arg) const;
		[[nodiscard]] llvm::StringRef getInverseFunction(llvm::StringRef invertibleArg) const;
		[[nodiscard]] llvm::ArrayRef<std::string> getInverseArgs(llvm::StringRef invertibleArg) const;
		void addInverse(llvm::StringRef invertedArg, llvm::StringRef inverseFunctionName, llvm::ArrayRef<std::string> args);

		private:
		llvm::StringMap<std::pair<std::string, Container<std::string>>> map;
	};
}
