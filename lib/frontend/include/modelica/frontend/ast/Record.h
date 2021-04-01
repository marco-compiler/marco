#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <modelica/utils/SourcePosition.h>
#include <memory>

namespace modelica::frontend
{
	class Member;

	class Record
	{
		private:
		template<typename T> using Container = llvm::SmallVector<std::shared_ptr<T>>;

		public:
		using iterator = boost::indirect_iterator<Container<Member>::iterator>;
		using const_iterator = boost::indirect_iterator<Container<Member>::const_iterator>;

		Record(SourcePosition location, std::string name, llvm::ArrayRef<Member> members);

		[[nodiscard]] bool operator==(const Record& other) const;
		[[nodiscard]] bool operator!=(const Record& other) const;

		[[nodiscard]] Member& operator[](llvm::StringRef name);
		[[nodiscard]] const Member& operator[](llvm::StringRef name) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] std::string& getName();
		[[nodiscard]] const std::string& getName() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin()const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		private:
		SourcePosition location;
		std::string name;
		Container<Member> members;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj);

	std::string toString(const Record& obj);
}
