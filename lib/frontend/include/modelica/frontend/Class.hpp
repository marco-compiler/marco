#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

#include "Algorithm.hpp"
#include "Equation.hpp"
#include "ForEquation.hpp"
#include "Func.hpp"
#include "Member.hpp"

namespace modelica
{
	class Class
	{
		public:
		Class(
				std::string name,
				llvm::ArrayRef<Member> members = {},
				llvm::ArrayRef<Equation> equations = {},
				llvm::ArrayRef<ForEquation> forEquations = {});

		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] llvm::SmallVectorImpl<Member>& getMembers();
		[[nodiscard]] llvm::SmallVectorImpl<Equation>& getEquations();
		[[nodiscard]] llvm::SmallVectorImpl<ForEquation>& getForEquations();
		[[nodiscard]] llvm::SmallVectorImpl<Algorithm>& getAlgorithms();
		[[nodiscard]] llvm::SmallVectorImpl<Func>& getFunctions();

		[[nodiscard]] const std::string& getName() const;
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getMembers() const;
		[[nodiscard]] const llvm::SmallVectorImpl<Equation>& getEquations() const;
		[[nodiscard]] const llvm::SmallVectorImpl<ForEquation>& getForEquations()
				const;
		[[nodiscard]] const llvm::SmallVectorImpl<Algorithm>& getAlgorithms() const;
		[[nodiscard]] const llvm::SmallVectorImpl<Func>& getFunctions() const;

		[[nodiscard]] size_t membersCount() const { return members.size(); }

		void addMember(Member newMember);
		void eraseMember(size_t memberIndex);
		void addEquation(Equation equation);
		void addForEquation(ForEquation equation);
		void addAlgorithm(Algorithm algorithm);
		void addFunction(Func function);

		private:
		std::string name;
		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<ForEquation, 3> forEquations;
		llvm::SmallVector<Algorithm, 3> algorithms;
		llvm::SmallVector<Func, 3> functions;
	};
}	 // namespace modelica
