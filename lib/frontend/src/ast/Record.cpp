#include <modelica/frontend/AST.h>
#include <numeric>

using namespace modelica;
using namespace frontend;

Record::Record(SourcePosition location,
							 std::string name,
							 llvm::ArrayRef<Member> members)
		: location(std::move(location)),
			name(std::move(name))
{
	for (const auto& member : members)
		this->members.emplace_back(std::make_shared<Member>(member));
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

Member& Record::operator[](llvm::StringRef memberName)
{
	return **std::find_if(members.begin(), members.end(),
												[&](const auto& member) { return member->getName() == memberName; });
}

const Member& Record::operator[](llvm::StringRef memberName) const
{
	return **std::find_if(members.begin(), members.end(),
												[&](const auto& member) { return member->getName() == memberName; });
}

void Record::dump() const
{
	dump(llvm::outs(), 0);
}

void Record::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "record: " << name << "\n";

	for (const auto& member : members)
		member->dump(os, indents + 1);
}

SourcePosition Record::getLocation() const
{
	return location;
}

std::string& Record::getName()
{
	return name;
}

const std::string& Record::getName() const
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

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Record& obj)
	{
		return "(" +
					 accumulate(obj.begin(), obj.end(), std::string(),
											[](const std::string& result, const Member& member)
											{
												std::string str = toString(member.getType()) + " " + member.getName();
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}
}
