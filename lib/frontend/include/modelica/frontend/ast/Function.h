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
		Function(SourceRange location,
						 bool pure,
						 llvm::StringRef name,
						 llvm::Optional<std::unique_ptr<Annotation>> annotation);

		Function(const Function& other);
		Function(Function&& other);

		~Function() override;

		Function& operator=(const Function& other);
		Function& operator=(Function&& other);

		friend void swap(Function& first, Function& second);

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] bool isPure() const;

		[[nodiscard]] bool hasAnnotation() const;
		[[nodiscard]] Annotation* getAnnotation();
		[[nodiscard]] const Annotation* getAnnotation() const;

		private:
		bool pure;
		std::string name;
		llvm::Optional<std::unique_ptr<Annotation>> annotation;
	};

	class DerFunction
			: public Function,
				public impl::Dumpable<DerFunction>
	{
		public:
		DerFunction(const DerFunction& other);
		DerFunction(DerFunction&& other);
		~DerFunction() override;

		DerFunction& operator=(const DerFunction& other);
		DerFunction& operator=(DerFunction&& other);

		friend void swap(DerFunction& first, DerFunction& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getDerivedFunction() const;
		[[nodiscard]] llvm::StringRef getArg() const;

		private:
		friend class Class;

		DerFunction(SourceRange location,
								bool pure,
								llvm::StringRef name,
								llvm::Optional<std::unique_ptr<Annotation>> annotation,
								llvm::StringRef derivedFunction,
								llvm::StringRef arg);

		std::string derivedFunction;
		std::string arg;
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

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Member>> getMembers();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Member>> getMembers() const;

		[[nodiscard]] Container<Member*> getArgs() const;
		[[nodiscard]] Container<Member*> getResults() const;
		[[nodiscard]] Container<Member*> getProtectedMembers() const;

		void addMember(std::unique_ptr<Member> member);

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Algorithm>> getAlgorithms();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Algorithm>> getAlgorithms() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type type);

		private:
		friend class Class;

		StandardFunction(SourceRange location,
										 bool pure,
										 llvm::StringRef name,
										 llvm::Optional<std::unique_ptr<Annotation>> annotation,
										 llvm::ArrayRef<std::unique_ptr<Member>> members,
										 llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms);

		Container<std::unique_ptr<Member>> members;
		Container<std::unique_ptr<Algorithm>> algorithms;
		Type type;
	};

	class InverseFunctionAnnotation
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		InverseFunctionAnnotation() = default;

		[[nodiscard]] bool isInvertible(llvm::StringRef arg) const;
		[[nodiscard]] llvm::StringRef getInverseFunction(llvm::StringRef invertibleArg) const;
		[[nodiscard]] llvm::ArrayRef<std::string> getInverseArgs(llvm::StringRef invertibleArg) const;
		void addInverse(llvm::StringRef invertedArg, llvm::StringRef inverseFunctionName, llvm::ArrayRef<std::string> args);

		private:
		llvm::StringMap<std::pair<std::string, Container<std::string>>> map;
	};
}
