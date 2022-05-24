#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "marco/ast/nodes/ASTNode.h"
#include <memory>

namespace marco::ast
{
	class Member;

	class Record
			: public ASTNode,
				public impl::Dumpable<Record>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Member>>::iterator;
		using const_iterator = Container<std::unique_ptr<Member>>::const_iterator;

		Record(const Record& other);
		Record(Record&& other);
		~Record() override;

		Record& operator=(const Record& other);
		Record& operator=(Record&& other);

		friend void swap(Record& first, Record& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] bool operator==(const Record& other) const;
		[[nodiscard]] bool operator!=(const Record& other) const;

		[[nodiscard]] Member* operator[](llvm::StringRef name);
		[[nodiscard]] const Member* operator[](llvm::StringRef name) const;

		[[nodiscard]] llvm::StringRef getName() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin()const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] bool shouldBeInlined() const;

		[[nodiscard]] const StandardFunction& getDefaultConstructor() const;
		
		void setAsNotInlineable();

		private:
		friend class Class;

		Record(SourceRange location,
					 llvm::StringRef name,
					 llvm::ArrayRef<std::unique_ptr<Member>> members);

		void setupDefaultConstructor();

		std::string name;
		Container<std::unique_ptr<Member>> members;

		std::unique_ptr<StandardFunction> defaultConstructor;
		bool inlineable=true;
	};

	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj);

	std::string toString(const Record& obj);

	class RecordInstance
		: public ASTNode,
		    public impl::Dumpable<RecordInstance>
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<std::unique_ptr<Expression>>::iterator;
		using const_iterator = Container<std::unique_ptr<Expression>>::const_iterator;

		RecordInstance(const RecordInstance& other);
		RecordInstance(RecordInstance&& other);
		~RecordInstance() override;

		RecordInstance& operator=(const RecordInstance& other);
		RecordInstance& operator=(RecordInstance&& other);

		[[nodiscard]] bool operator==(const RecordInstance& other) const;
		[[nodiscard]] bool operator!=(const RecordInstance& other) const;

		[[nodiscard]] bool isLValue() const;

		[[nodiscard]] Type& getType();
		[[nodiscard]] const Type& getType() const;
		void setType(Type tp);

		[[nodiscard]] const Record* getRecordType() const;

		friend void swap(RecordInstance& first, RecordInstance& second);

		void print(llvm::raw_ostream& os, size_t indents = 0) const override;

		[[nodiscard]] Expression& getMemberValue(llvm::StringRef name);


		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin()const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;


		private:
		friend class Expression;

		RecordInstance(SourceRange location,
					   		Type type,
							llvm::ArrayRef<std::unique_ptr<Expression>> values);

		const Record *recordType;
		Type type;
		Container<std::unique_ptr<Expression>> values;

	};
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const RecordInstance& obj);

	std::string toString(const RecordInstance& obj);
}
