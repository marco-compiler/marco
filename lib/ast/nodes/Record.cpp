#include "marco/ast/AST.h"
#include <numeric>

using namespace marco::ast;

Record::Record(SourceRange location,
							 llvm::StringRef name,
							 llvm::ArrayRef<std::unique_ptr<Member>> members)
		: ASTNode(std::move(location)),
			name(name.str())
{
	for (const auto& member : members)
		this->members.push_back(member->clone());
	
	setupDefaultConstructor();
}

Record::Record(const Record& other)
		: ASTNode(other),
			name(other.name)
{
	for (const auto& member : other.members)
		this->members.push_back(member->clone());
	
	setupDefaultConstructor();
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

/**
 * default constructor is inline, accepts one param for each member and return a record
 * 
 * Inline property AST structure:
 *
 *          class-modification
 *                  |
 *            argument-list
 *             /         \
 *       argument         ...
 *          |
 *  element-modification
 *    /           \
 *  name        modification
 * inline           |
 *              expression
 *                true
 */
void Record::setupDefaultConstructor()
{
	auto loc = getLocation();
	llvm::SmallVector<std::unique_ptr<Algorithm>, 3> algorithms;


	llvm::SmallVector<std::unique_ptr<Argument>, 3> arguments;


	auto mod = Modification::build(std::move(loc), Expression::constant(loc,Type(BuiltInType::Boolean),true));
	arguments.push_back(Argument::elementModification(loc, false, false, "inline", std::move(mod)));
	auto class_mod = ClassModification::build(loc, arguments);
	
	std::unique_ptr<Annotation> annotation = std::make_unique<Annotation>(std::move(loc),std::move(class_mod));
	llvm::Optional<std::unique_ptr<Annotation>> clsAnnotation = std::move(annotation);

	defaultConstructor = StandardFunction::build(
				getLocation(),
				true,
				getName(),
				members,
				algorithms,
				std::move(clsAnnotation)
			);
}

namespace marco::ast
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

bool Record::shouldBeInlined() const
{
	//is true iff all functions that use it are inline-able
	return inlineable;
}

void Record::setAsNotInlineable()
{
	inlineable = false;
}

const StandardFunction& Record::getDefaultConstructor() const
{
	return *defaultConstructor;
}





RecordInstance::RecordInstance(SourceRange location, Type type, llvm::ArrayRef<std::unique_ptr<Expression>> values)
	: ASTNode(std::move(location)),type(type)
{
	setType(std::move(type));

	for (const auto& v : values)
		this->values.push_back(v->clone());
}

RecordInstance::RecordInstance(const RecordInstance& other)
	: ASTNode(other),
		recordType(other.recordType),
		type(other.type)
{
	for (const auto& v : other.values)
		values.push_back(v->clone());
}

RecordInstance::RecordInstance(RecordInstance&& other) = default;
RecordInstance::~RecordInstance()  = default;

RecordInstance& RecordInstance::operator=(const RecordInstance& other)
{
	RecordInstance result(other);
	swap(*this, result);
	return *this;
}
RecordInstance& RecordInstance::operator=(RecordInstance&& other) = default;

bool RecordInstance::operator==(const RecordInstance& other) const
{
	if (recordType != other.recordType)
		return false;

	if (values.size() != other.values.size())
		return false;

	auto pairs = llvm::zip(values, other.values);
	return std::all_of(pairs.begin(), pairs.end(),
										 [](const auto& pair)
										 {
											 const auto& [x, y] = pair;
											 return *x == *y;
										 });
}

bool RecordInstance::operator!=(const RecordInstance& other) const
{
	return !(*this == other);
}

bool RecordInstance::isLValue() const
{
	return false;
}

void RecordInstance::setType(Type tp)
{
	assert(type.isa<Record*>());
	assert(type.get<Record*>()!=nullptr);

	type = std::move(tp);
	recordType = type.get<Record*>();
}

Type& RecordInstance::getType()
{
	return type;
}

const Type& RecordInstance::getType() const
{
	return type;
}

const Record* RecordInstance::getRecordType() const
{
	return recordType;
}

RecordInstance::iterator RecordInstance::begin()
{
	return values.begin();
}

RecordInstance::const_iterator RecordInstance::begin() const
{
	return values.begin();
}

RecordInstance::iterator RecordInstance::end()
{
	return values.end();
}

RecordInstance::const_iterator RecordInstance::end() const
{
	return values.end();
}

namespace marco::ast
{
	void swap(RecordInstance& first, RecordInstance& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		impl::swap(first.values, second.values);
		swap(first.type, second.type);
		swap(first.recordType, second.recordType);
	}
}

void RecordInstance::print(llvm::raw_ostream& os, size_t indents) const 
{
	os.indent(indents);
	os << "record instance (" << recordType->getName() << ") : \n";

	for (const auto& v : values)
		v->print(os, indents + 1);
}

Expression& RecordInstance::getMemberValue(llvm::StringRef name)
{
	auto it = values.begin();

	for(const auto& m : *recordType){
		if(m->getName() == name){
			return **it;
		}
		++it;
	}
	assert(false && "not enough values in record instance");
	return **it;
}

namespace marco::ast
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Record& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Record& obj)
	{
		return obj.getName().str() + "(" +
					 accumulate(obj.begin(), obj.end(), std::string(),
											[](const std::string& result, const std::unique_ptr<Member>& member)
											{
												std::string str = toString(member->getType()) + " " + member->getName().str();
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}


	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const RecordInstance& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const RecordInstance& obj)
	{
		return "RecordInstance<" + 
					obj.getRecordType()->getName().str() + ">(" +
					 accumulate(obj.begin(), obj.end(), std::string(),
											[](const std::string& result, const std::unique_ptr<Expression>& e)
											{
												std::string str = toString(*e);
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}
}
