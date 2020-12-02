#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/frontend/Algorithm.hpp>
#include <modelica/frontend/Equation.hpp>
#include <modelica/frontend/ForEquation.hpp>
#include <modelica/frontend/Member.hpp>
#include <string>

namespace modelica
{
	class ClassContainer;

	class Class
	{
		private:
		using InnerClass = std::unique_ptr<ClassContainer>;
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		Class(
				std::string name,
				llvm::ArrayRef<Member> members,
				llvm::ArrayRef<Equation> equations,
				llvm::ArrayRef<ForEquation> forEquations,
				llvm::ArrayRef<Algorithm> algorithms,
				llvm::ArrayRef<ClassContainer> innerClasses);

		Class(const Class& other);
		Class(Class&& other) = default;

		Class& operator=(const Class& other);
		Class& operator=(Class&& other) = default;

		~Class() = default;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] std::string getName() const;

		[[nodiscard]] Container<Member>& getMembers();
		[[nodiscard]] const Container<Member>& getMembers() const;
		void addMember(Member member);

		[[nodiscard]] Container<Equation>& getEquations();
		[[nodiscard]] const Container<Equation>& getEquations() const;

		[[nodiscard]] Container<ForEquation>& getForEquations();
		[[nodiscard]] const Container<ForEquation>& getForEquations() const;

		[[nodiscard]] Container<Algorithm>& getAlgorithms();
		[[nodiscard]] const Container<Algorithm>& getAlgorithms() const;

		[[nodiscard]] Container<InnerClass>& getInnerClasses();
		[[nodiscard]] const Container<InnerClass>& getInnerClasses() const;

		private:
		std::string name;
		Container<Member> members;
		Container<Equation> equations;
		Container<ForEquation> forEquations;
		Container<Algorithm> algorithms;
		Container<InnerClass> innerClasses;
	};
}	 // namespace modelica
