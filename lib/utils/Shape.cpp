#include "marco/utils/Shape.h"
#include <vector>
#include "marco/utils/IRange.hpp"


namespace marco
{


Shape::DimensionSize::DimensionSize():value(Undefined()){}
Shape::DimensionSize::DimensionSize(long int v):value(v){}
Shape::DimensionSize::DimensionSize(std::initializer_list<DimensionSize> ragged):value(std::make_unique<llvm::SmallVector<DimensionSize,3>>(ragged)){}


Shape::DimensionSize::DimensionSize(const Shape::DimensionSize::Container<Shape::DimensionSize> &ragged)
    :value(std::make_unique<Shape::DimensionSize::Container<Shape::DimensionSize>>(ragged)){}
    
Shape::DimensionSize::DimensionSize(std::unique_ptr<Container<Shape::DimensionSize>> ragged)
    :value(std::move(ragged)){}

Shape::DimensionSize::DimensionSize(const Shape::DimensionSize& other){
    if (other.isUndefined())
		value = DimensionSize::Undefined();
	else if (other.isRagged())
		value = std::make_unique< Container<Shape::DimensionSize> >(*std::get< std::unique_ptr<Container<Shape::DimensionSize>> >(other.value));
	else
		value = other.getNumericValue();
}
Shape::DimensionSize::DimensionSize(Shape::DimensionSize&& other) = default;
Shape::DimensionSize::~DimensionSize() = default;

Shape::DimensionSize& Shape::DimensionSize::operator=(const Shape::DimensionSize& other){
    Shape::DimensionSize result(other);
	swap(*this, result);
	return *this;
}
Shape::DimensionSize& Shape::DimensionSize::operator=(Shape::DimensionSize&& other) = default;

void swap(Shape::DimensionSize& first, Shape::DimensionSize& second){
    std::swap(first.value,second.value);
}

bool Shape::DimensionSize::isRagged() const
{
	return std::holds_alternative<Ragged>(value);
}
bool Shape::DimensionSize::isUndefined() const
{
	return std::holds_alternative<Undefined>(value);
}
bool Shape::DimensionSize::isConstant() const
{
	return !isUndefined();
}

template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// Shape::DimensionSize::operator long() const
// {
//     assert(!isRagged());
//     // if(isRagged())return -2;
//     if(isUndefined())return -1;
    
//     return std::get<long>(value);
// }

long Shape::DimensionSize::getNumericValue() const
{
    assert(!isRagged());
    // if(isRagged())return -2;
    if(isUndefined())return -1;
    
    return std::get<long>(value);    
}

llvm::ArrayRef<Shape::DimensionSize> Shape::DimensionSize::asRagged() const
{
    return *(std::get<Ragged>(value));
}

bool Shape::DimensionSize::operator==(const DimensionSize& other) const
{
    if(isRagged()){
        if(other.isRagged()){
            return asRagged()==other.asRagged();
        }
        return false;
    }
    return value==other.value;
}
bool Shape::DimensionSize::operator!=(const DimensionSize& other) const
{
    return !((*this)==other);
}

llvm::hash_code hash_value(const Shape::DimensionSize& d)
{
    if(d.isRagged())return -2;//todo: hashing for ragged

    return d.getNumericValue();
}
            

Shape::iterator Shape::begin()
{
	return sizes.begin();
}
Shape::iterator Shape::end()
{
	return sizes.end();
}

Shape::const_iterator Shape::begin() const
{
	return sizes.begin();
}
Shape::const_iterator Shape::end() const
{
	return sizes.end();
}

Shape::DimensionSize& Shape::operator[](int index)
{
    return sizes[index];
}

const Shape::DimensionSize& Shape::operator[](int index) const
{
    return sizes[index];
}

bool Shape::empty() const
{
	return sizes.empty();
}

size_t Shape::size() const
{
	return sizes.size();
}

void Shape::push_back(const DimensionSize& d)
{
    sizes.push_back(d);
}

bool Shape::operator==(const Shape& other) const
{
    return sizes==other.sizes;
}
bool Shape::operator!=(const Shape& other) const
{
    return !(*this==other);
}

llvm::SmallVector<long, 3> Shape::to_old() const
{
    llvm::SmallVector<long, 3> v;

    for(auto a:sizes){
        v.push_back(a.getNumericValue());
    }
    return v;
}
// Shape::operator llvm::SmallVector<long,3>() const
// {
//     return to_old();
// }

bool Shape::isRagged() const
{
	return std::find_if(sizes.begin(), sizes.end(), [](const DimensionSize &s){return s.isRagged();}) != sizes.end();
}
bool Shape::isUndefined() const
{
    return std::find_if(sizes.begin(), sizes.end(), [](const DimensionSize& s){return s.isUndefined();}) != sizes.end();
}
bool Shape::isConstant() const
{
    return !isUndefined();
}

// Shape::Shape(std::initializer_list<long int> values)
// {
//     for(auto v:values)
//     {
//         if(v==-1)sizes.emplace_back();
//         else sizes.push_back(v);
//     }
// }

Shape::Shape(std::initializer_list<Shape::DimensionSize> values)
{
    for(auto v:values)
    {
        // if(v==-1)sizes.emplace_back();
        // else 
        sizes.push_back(v);
    }
}

Shape::Shape(llvm::ArrayRef<long> values)
{
    for(auto v:values)
    {
        if(v==-1)sizes.emplace_back();
        else sizes.push_back(v);
    }
}
Shape::Shape(const llvm::SmallVector<long,3> &values)
{
    for(auto v:values)
    {
        if(v==-1)sizes.emplace_back();
        else sizes.push_back(v);
    }
}

Shape::Shape(llvm::NoneType)
{

}
Shape::Shape(Shape::DimensionSize el)
{
    push_back(el);
}
Shape::Shape(Shape::DimensionSize el,size_t length)
{
    while(length--)
        push_back(el);
}

llvm::ArrayRef<Shape::DimensionSize> Shape::dimensions() const
{
    return sizes;
}


std::string toString(const Shape::DimensionSize &dim){

    if(dim.isRagged()){
        std::string s;
        std::string padding = "";
        for (const auto& val : dim.asRagged()) {
            s += padding + toString(val);
            padding = ", ";
        }
        return "{"+s+"}";
    }
    
    return std::to_string(dim.getNumericValue());
}

std::string toString(const Shape &shape){
    std::string s;
    std::string padding = "";

    for(auto &it:shape){
        s += padding + toString(it);
        padding = ", ";
    }
    return "["+s+"]";
}



long getCurrentRaggedDimension(const std::vector<long> &partialIndex, const Shape::DimensionSize &dim, size_t ragged_depth=0)
{
	if(!dim.isRagged())return dim.getNumericValue();
	
	assert(partialIndex.size());

	long index = partialIndex[partialIndex.size()-ragged_depth];

	return getCurrentRaggedDimension(partialIndex,dim.asRagged()[index], ragged_depth-1);
}


std::vector<std::vector<long>> listIndexes(std::vector<long> partialIndex, llvm::ArrayRef<Shape::DimensionSize> nextDimensions, size_t ragged_depth=0)
{
    if(nextDimensions.empty())return {partialIndex};

    auto dim = nextDimensions[0];

    std::vector<std::vector<long>> result;
	
    // ragged_depth is used to keep track of the ragged dimension branching point:
    // it remains 0 in the non-ragged case
    // when a ragged dimension is encountered it's setted to 1 and it's increased at every next dimension.
	ragged_depth = ragged_depth ? ragged_depth+1 : (dim.isRagged() ? 1 : 0);

	for(auto it:irange(getCurrentRaggedDimension(partialIndex,dim,ragged_depth)))
	{
		auto p = partialIndex;
		p.push_back(it);
		auto r = listIndexes(p,nextDimensions.slice(1),ragged_depth);
		result.insert(result.end(),r.begin(),r.end());
	}

    return result;
}

std::vector<std::vector<long>> generateAllIndexes(Shape shape)
{
    return listIndexes({},shape.dimensions());
}

}