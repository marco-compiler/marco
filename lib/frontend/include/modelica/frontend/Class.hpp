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
	using UniqueClass = std::unique_ptr<ClassContainer>;

	class Class
	{
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

		[[nodiscard]] llvm::SmallVectorImpl<Member>& getMembers();
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getMembers() const;
		void addMember(Member member);

		[[nodiscard]] llvm::SmallVectorImpl<Equation>& getEquations();
		[[nodiscard]] const llvm::SmallVectorImpl<Equation>& getEquations() const;

		[[nodiscard]] llvm::SmallVectorImpl<ForEquation>& getForEquations();
		[[nodiscard]] const llvm::SmallVectorImpl<ForEquation>& getForEquations()
		const;

		[[nodiscard]] llvm::SmallVectorImpl<Algorithm>& getAlgorithms();
		[[nodiscard]] const llvm::SmallVectorImpl<Algorithm>& getAlgorithms() const;

		[[nodiscard]] llvm::SmallVectorImpl<UniqueClass>& getInnerClasses();
		[[nodiscard]] const llvm::SmallVectorImpl<UniqueClass>& getInnerClasses() const;

		private:
		std::string name;
		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Equation, 3> equations;
		llvm::SmallVector<ForEquation, 3> forEquations;
		llvm::SmallVector<Algorithm, 3> algorithms;
		llvm::SmallVector<UniqueClass, 3> innerClasses;
	};
}	 // namespace modelica
