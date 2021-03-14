#pragma once

#include <boost/range/join.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourceRange.hpp>
#include <string>

#include "Algorithm.h"
#include "Member.h"

namespace modelica
{
	class Function
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>, 3>;

		public:
		Function(SourcePosition location,
						 std::string name,
						 bool pure,
						 llvm::ArrayRef<Member> members,
						 llvm::ArrayRef<Algorithm> algorithms);

		Member& operator[](llvm::StringRef name);
		const Member& operator[](llvm::StringRef name) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] bool isPure() const;

		[[nodiscard]] Container<Member>& getMembers();
		[[nodiscard]] const Container<Member>& getMembers() const;

		[[nodiscard]] Container<Member> getArgs() const;
		[[nodiscard]] Container<Member> getResults() const;
		[[nodiscard]] Container<Member> getProtectedMembers() const;

		void addMember(Member member);

		[[nodiscard]] Container<Algorithm>& getAlgorithms();
		[[nodiscard]] const Container<Algorithm>& getAlgorithms() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type type);

		private:
		SourcePosition location;
		std::string name;
		bool pure;
		Container<Member> members;
		Container<Algorithm> algorithms;
		Type type;
	};
}	 // namespace modelica
