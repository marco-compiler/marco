#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourceRange.hpp>
#include <string>

namespace modelica
{
	class Member;
	class Equation;
	class ForEquation;
	class Algorithm;
	class ClassContainer;

	class Class
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		Class(
				SourcePosition location,
				std::string name,
				llvm::ArrayRef<Member> members,
				llvm::ArrayRef<Equation> equations,
				llvm::ArrayRef<ForEquation> forEquations,
				llvm::ArrayRef<Algorithm> algorithms,
				llvm::ArrayRef<ClassContainer> innerClasses);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] Container<Member>& getMembers();
		[[nodiscard]] const Container<Member>& getMembers() const;
		void addMember(Member member);

		[[nodiscard]] Container<Equation>& getEquations();
		[[nodiscard]] const Container<Equation>& getEquations() const;

		[[nodiscard]] Container<ForEquation>& getForEquations();
		[[nodiscard]] const Container<ForEquation>& getForEquations() const;

		[[nodiscard]] Container<Algorithm>& getAlgorithms();
		[[nodiscard]] const Container<Algorithm>& getAlgorithms() const;

		[[nodiscard]] Container<ClassContainer>& getInnerClasses();
		[[nodiscard]] const Container<ClassContainer>& getInnerClasses() const;

		private:
		SourcePosition location;
		std::string name;
		Container<Member> members;
		Container<Equation> equations;
		Container<ForEquation> forEquations;
		Container<Algorithm> algorithms;
		Container<ClassContainer> innerClasses;
	};
}	 // namespace modelica
