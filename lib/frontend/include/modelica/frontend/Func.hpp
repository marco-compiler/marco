#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <string>

#include "Algorithm.hpp"
#include "Member.hpp"

namespace modelica
{
	class Func
	{
		public:
		Func(
				std::string name,
				llvm::ArrayRef<Member> input = {},
				llvm::ArrayRef<Member> output = {},
				llvm::ArrayRef<Member> members = {},
				llvm::ArrayRef<Algorithm> algorithms = {});

		void dump(llvm::raw_ostream& os = llvm::outs(), size_t indents = 0) const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] llvm::SmallVectorImpl<Member>& getInputVariables();
		[[nodiscard]] llvm::SmallVectorImpl<Member>& getOutputVariables();
		[[nodiscard]] llvm::SmallVectorImpl<Member>& getMembers();
		[[nodiscard]] llvm::SmallVectorImpl<Algorithm>& getAlgorithms();

		[[nodiscard]] const std::string& getName() const;
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getInputVariables()
				const;
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getOutputVariables()
				const;
		[[nodiscard]] const llvm::SmallVectorImpl<Member>& getMembers() const;
		[[nodiscard]] const llvm::SmallVectorImpl<Algorithm>& getAlgorithms() const;

		private:
		std::string name;
		llvm::SmallVector<Member, 3> input;
		llvm::SmallVector<Member, 3> output;
		llvm::SmallVector<Member, 3> members;
		llvm::SmallVector<Algorithm, 3> algorithms;
	};
}	 // namespace modelica
