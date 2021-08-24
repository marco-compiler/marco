#include <marco/frontend/AST.h>
#include <numeric>

using namespace marco::frontend;

Record::Record(SourceRange location,
							 llvm::StringRef name,
							 llvm::ArrayRef<std::unique_ptr<Member>> members)
		: ASTNode(std::move(location)),
			name(name.str())
{
	for (const auto& member : members)
		this->members.push_back(member->clone());
}

Record::Record(const Record& other)
		: ASTNode(other),
			name(other.name)
{
	for (const auto& member : other.members)
		this->members.push_back(member->clone());
}

Record::Record(Record&& other) = default;

Record::~Record() = default;

Record& Record::operator=(const Record& other)
{
	Record result(other);
	swap(*this, result);
	return *this;
}

Record& Record::operator=(Record&& other) = default;

namespace marco::frontend
{
	void swap(Record& first, Record& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		impl::swap(first.members, second.members);
	}
}

bool Record::operator==(const Record& other) const
{
	if (name != other.name)
		return false;

	if (members.size() != other.members.size())
		return false;

	auto pairs = llvm::zip(members, other.members);
	return std::all_of(pairs.begin(), pairs.end(),
										 [](const auto& pair)
										 {
											 const auto& [x, y] = pair;
											 return *x == *y;
										 });
}

bool Record::operator!=(const Record& other) const
{
	return !(*this == other);
}

Member* Record::operator[](llvm::StringRef name)
{
	return std::find_if(
			members.begin(), members.end(),
			[&](const auto& member) {
				return member->getName() == name;
			})->get();
}

const Member* Record::operator[](llvm::StringRef name) const
{
	return std::find_if(
			members.begin(), members.end(),
			[&](const auto& member) {
				return member->getName() == name;
			})->get();
}

void Record::print(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "record: " << name << "\n";

	for (const auto& member : members)
		member->print(os, indents + 1);
}

llvm::StringRef Record::getName() const
{
	return name;
}

size_t Record::size() const
{
	return members.size();
}

Record::iterator Record::begin()
{
	return members.begin();
}

Record::const_iterator Record::begin() const
{
	return members.begin();
}

Record::iterator Record::end()
{
	return members.end();
}

Record::const_iterator Record::end() const
{
	return members.end();
}

namespace marco::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Record& obj)
	{
		return "(" +
					 accumulate(obj.begin(), obj.end(), std::string(),
											[](const std::string& result, const std::unique_ptr<Member>& member)
											{
												std::string str = toString(member->getType()) + " " + member->getName().str();
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}
}
