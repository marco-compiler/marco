#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/frontend/Algorithm.hpp>
#include <modelica/frontend/Member.hpp>
#include <string>

namespace modelica
{
	class Function
	{
		public:
		Function(std::string name,
						 bool pure,
						 llvm::ArrayRef<Member> members,
						 llvm::ArrayRef<Algorithm> algorithms);

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] std::string getName() const;

		[[nodiscard]] llvm::SmallVectorImpl<Member>& getMembers();
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getMembers() const;
		void addMember(Member member);

		[[nodiscard]] llvm::SmallVectorImpl<Algorithm>& getAlgorithms();
		[[nodiscard]] const llvm::SmallVectorImpl<Algorithm>& getAlgorithms() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type type);

		private:
		std::string name;
		bool pure;
		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Algorithm, 3> algorithms;
		Type type;
	};
}	 // namespace modelica
