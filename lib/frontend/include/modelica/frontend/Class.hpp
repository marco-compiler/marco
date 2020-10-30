#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

#include "Algorithm.hpp"
#include "Equation.hpp"
#include "ForEquation.hpp"
#include "Member.hpp"

namespace modelica
{
	enum class ClassType
	{
		Block,
		Class,
		Connector,
		Function,
		Model,
		Package,
		Operator,
		Record,
		Type
	};

	std::string toString(ClassType type);

	class Class;

	using Func = std::unique_ptr<Class>;

	class Class
	{
		public:
		Class(
				ClassType type,
				std::string name,
				llvm::ArrayRef<Member> members = {},
				llvm::ArrayRef<Equation> equations = {},
				llvm::ArrayRef<ForEquation> forEquations = {});

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		template<ClassType T>
		bool isA()
		{
			return type == T;
		}

		[[nodiscard]] ClassType getType() const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] llvm::SmallVectorImpl<Member>& getMembers();
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getMembers() const;
		[[nodiscard]] size_t membersCount() const;

		[[nodiscard]] llvm::SmallVectorImpl<Equation>& getEquations();
		[[nodiscard]] const llvm::SmallVectorImpl<Equation>& getEquations() const;

		[[nodiscard]] llvm::SmallVectorImpl<ForEquation>& getForEquations();
		[[nodiscard]] const llvm::SmallVectorImpl<ForEquation>& getForEquations()
				const;

		[[nodiscard]] llvm::SmallVectorImpl<Algorithm>& getAlgorithms();
		[[nodiscard]] const llvm::SmallVectorImpl<Algorithm>& getAlgorithms() const;

		[[nodiscard]] llvm::SmallVectorImpl<Func>& getFunctions();
		[[nodiscard]] const llvm::SmallVectorImpl<Func>& getFunctions() const;

		void addMember(Member newMember);
		void eraseMember(size_t memberIndex);
		void addEquation(Equation equation);
		void addForEquation(ForEquation equation);
		void addAlgorithm(Algorithm algorithm);
		void addFunction(Class function);

		private:
		ClassType type;
		std::string name;
		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<ForEquation, 3> forEquations;
		llvm::SmallVector<Algorithm, 3> algorithms;
		llvm::SmallVector<Func, 3> functions;
	};
}	 // namespace modelica
