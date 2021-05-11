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

	class Function : public Class
	{
		public:
		Function(ASTNodeKind kind,
						 SourcePosition location,
						 bool pure,
						 llvm::StringRef name,
						 llvm::Optional<std::unique_ptr<Annotation>>& annotation);

		Function(const Function& other);
		Function(Function&& other);

		~Function() override;

		Function& operator=(const Function& other);
		Function& operator=(Function&& other);

		friend void swap(Function& first, Function& second);

		[[maybe_unused]] static bool classof(const ASTNode* node)
		{
			return node->getKind() >= ASTNodeKind::FUNCTION &&
						 node->getKind() <= ASTNodeKind::FUNCTION_LAST;
		}

		[[nodiscard]] virtual std::unique_ptr<Function> cloneFunction() const = 0;

		[[nodiscard]] bool isPure() const;

		[[nodiscard]] bool hasAnnotation() const;
		[[nodiscard]] Annotation* getAnnotation();
		[[nodiscard]] const Annotation* getAnnotation() const;

		private:
		bool pure;
		llvm::Optional<std::unique_ptr<Annotation>> annotation;
	};

	namespace impl
	{
		template<typename Derived>
		struct FunctionCRTP : public Function
		{
			using Function::Function;

			[[nodiscard]] std::unique_ptr<Class> cloneClass() const override
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}

			[[nodiscard]] std::unique_ptr<Function> cloneFunction() const override
			{
				return std::make_unique<Derived>(static_cast<const Derived&>(*this));
			}
		};
	}

	class DerFunction : public impl::FunctionCRTP<DerFunction>
	{
		public:
		DerFunction(SourcePosition location,
								bool pure,
								llvm::StringRef name,
								llvm::Optional<std::unique_ptr<Annotation>>& annotation,
								llvm::StringRef derivedFunction,
								llvm::StringRef arg);

		DerFunction(const DerFunction& other);
		DerFunction(DerFunction&& other);
		~DerFunction() override;

		DerFunction& operator=(const DerFunction& other);
		DerFunction& operator=(DerFunction&& other);

		friend void swap(DerFunction& first, DerFunction& second);

		static bool classof(const Function* obj)
		{
			return obj->getKind() == ASTNodeKind::FUNCTION_DER;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] llvm::StringRef getDerivedFunction() const;
		[[nodiscard]] llvm::StringRef getArg() const;

		private:
		std::string derivedFunction;
		std::string arg;
	};

	class StandardFunction : public impl::FunctionCRTP<StandardFunction>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		StandardFunction(SourcePosition location,
										 bool pure,
										 llvm::StringRef name,
										 llvm::Optional<std::unique_ptr<Annotation>>& annotation,
										 llvm::ArrayRef<std::unique_ptr<Member>> members,
										 llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms);

		StandardFunction(const StandardFunction& other);
		StandardFunction(StandardFunction&& other);
		~StandardFunction() override;

		StandardFunction& operator=(const StandardFunction& other);
		StandardFunction& operator=(StandardFunction&& other);

		friend void swap(StandardFunction& first, StandardFunction& second);

		static bool classof(const Function* obj)
		{
			return obj->getKind() == ASTNodeKind::FUNCTION_STANDARD;
		}

		void dump(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Member* operator[](llvm::StringRef name);
		[[nodiscard]] const Member* operator[](llvm::StringRef name) const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Member>> getMembers();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Member>> getMembers() const;

		[[nodiscard]] Container<Member*> getArgs() const;
		[[nodiscard]] Container<Member*> getResults() const;
		[[nodiscard]] Container<Member*> getProtectedMembers() const;

		void addMember(Member* member);

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Algorithm>> getAlgorithms();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Algorithm>> getAlgorithms() const;

		[[nodiscard]] Type getType() const;
		void setType(Type type);

		private:
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
