#include "marco/model/ModConst.hpp"
#include "marco/model/ModType.hpp"
#include <cmath>

using namespace marco;
using namespace std;
using namespace llvm;

template<typename L, typename R, typename Callable>
static ModConst::Content<L> contentPairWiseOperation(
		const ModConst::Content<L>& l, const ModConst::Content<R>& r, Callable&& c)
{
	ModConst::Content<L> out;
	for (int num : irange(l.size()))
		out.emplace_back(c(l[num], r[num]));

	return out;
}
template<typename L, typename Callable>
static ModConst partialPairWiseOperation(
		const ModConst::Content<L>& lContent, const ModConst& r, Callable&& c)
{
	return r.map([&](const auto& rContent) {
		return contentPairWiseOperation(lContent, rContent, c);
	});
}

template<typename Callable>
static ModConst pairWiseOperation(
		const ModConst& l, const ModConst& r, Callable&& c)
{
	return l.map([&](const auto& lContent) {
		return partialPairWiseOperation(lContent, r, c);
	});
}

ModConst ModConst::sum(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l + r; });
}

ModConst ModConst::sub(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l - r; });
}

ModConst ModConst::mult(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l * r; });
}

ModConst ModConst::divide(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l / r; });
}

ModConst ModConst::greaterThan(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l > r; })
			.as<bool>();
}

ModConst ModConst::greaterEqual(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l >= r; })
			.as<bool>();
}

ModConst ModConst::equal(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l == r; })
			.as<bool>();
}
ModConst ModConst::different(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l != r; })
			.as<bool>();
}

ModConst ModConst::lessThan(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l < r; })
			.as<bool>();
}

ModConst ModConst::lessEqual(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(left, right, [](auto l, auto r) { return l <= r; })
			.as<bool>();
}

ModConst ModConst::elevate(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	return pairWiseOperation(
			left, right, [](auto l, auto r) { return pow(l, r); });
}

ModConst ModConst::module(const ModConst& left, const ModConst& right)
{
	assert(left.size() == right.size());
	assert(left.getBuiltinType() == BultinModTypes::INT);
	assert(right.getBuiltinType() == BultinModTypes::INT);
	return contentPairWiseOperation<long, long>(
			left.getContent<long>(), right.getContent<long>(), [](long l, long r) {
				return l % r;
			});
}

template<typename T>
BultinModTypes modTypeFromContentType(const ModConst::Content<T>& content)
{
	return typeToBuiltin<T>();
}

BultinModTypes ModConst::getBuiltinType() const
{
	return map(
			[](const auto& content) { return modTypeFromContentType(content); });
}
