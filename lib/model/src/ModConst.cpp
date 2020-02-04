#include "modelica/model/ModConst.hpp"

#include "modelica/model/ModType.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

template<typename L, typename R, typename Callable>
static ModConst::Content<L> contentPairWiseOperation(
		const ModConst::Content<L>& l, const ModConst::Content<R>& r, Callable&& c)
{
	ModConst::Content<L> out;
	for (int num : irange(l.size()))
		out.push_back(c(l[num], r[num]));

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
