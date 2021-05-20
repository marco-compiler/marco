#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <string>

#include "Class.h"

namespace modelica::frontend
{
	class Algorithm;
	class Class;
	class Equation;
	class ForEquation;
	class Member;

	enum class ClassType
	{
		Function,
		Model,
		Package,
		Record
	};

	class Model
			: public ASTNode,
				public impl::Dumpable<Model>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		Model(const Model& other);
		Model(Model&& other);
		~Model() override;

		Model& operator=(const Model& other);
		Model& operator=(Model&& other);

		friend void swap(Model& first, Model& second);

		void print(llvm::raw_ostream& os, size_t indents) const override;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Member>> getMembers();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Member>> getMembers() const;
		void addMember(std::unique_ptr<Member> member);

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Equation>> getEquations();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Equation>> getEquations() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<ForEquation>> getForEquations();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<ForEquation>> getForEquations() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Algorithm>> getAlgorithms();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Algorithm>> getAlgorithms() const;

		[[nodiscard]] llvm::MutableArrayRef<std::unique_ptr<Class>> getInnerClasses();
		[[nodiscard]] llvm::ArrayRef<std::unique_ptr<Class>> getInnerClasses() const;

		private:
		friend class Class;

		Model(SourceRange location,
					llvm::StringRef name,
					llvm::ArrayRef<std::unique_ptr<Member>> members,
					llvm::ArrayRef<std::unique_ptr<Equation>> equations,
					llvm::ArrayRef<std::unique_ptr<ForEquation>> forEquations,
					llvm::ArrayRef<std::unique_ptr<Algorithm>> algorithms,
					llvm::ArrayRef<std::unique_ptr<Class>> innerClasses);

		std::string name;
		Container<std::unique_ptr<Member>> members;
		Container<std::unique_ptr<Equation>> equations;
		Container<std::unique_ptr<ForEquation>> forEquations;
		Container<std::unique_ptr<Algorithm>> algorithms;
		Container<std::unique_ptr<Class>> innerClasses;
	};
}
