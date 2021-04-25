#pragma once

#include <boost/range/join.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourcePosition.h>
#include <string>

#include "Algorithm.h"
#include "Annotation.h"
#include "Member.h"

namespace modelica::frontend
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
						 llvm::ArrayRef<Algorithm> algorithms,
						 Annotation annotation = Annotation());

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

		[[nodiscard]] Annotation getAnnotation() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type type);

		private:
		SourcePosition location;
		std::string name;
		bool pure;
		Container<Member> members;
		Container<Algorithm> algorithms;
		Annotation annotation;
		Type type;
	};

	class InverseFunctionAnnotation
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		InverseFunctionAnnotation()= default;

		[[nodiscard]] bool isInvertible(llvm::StringRef arg) const;
		[[nodiscard]] llvm::StringRef getInverseFunction(llvm::StringRef invertibleArg) const;
		[[nodiscard]] llvm::ArrayRef<std::string> getInverseArgs(llvm::StringRef invertibleArg) const;
		void addInverse(llvm::StringRef invertedArg, llvm::StringRef inverseFunctionName, llvm::ArrayRef<std::string> args);

		private:
		llvm::StringMap<std::pair<std::string, Container<std::string>>> map;
	};
}
