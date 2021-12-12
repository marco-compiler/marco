#include <llvm/ADT/STLExtras.h>
#include <marco/matching/Point.h>

using namespace marco::matching;

Point::Point(Point::data_type value)
{
  values.push_back(std::move(value));
}

Point::Point(std::initializer_list<Point::data_type> values) : values(std::move(values))
{
}

Point::Point(llvm::ArrayRef<Point::data_type> values) : values(values.begin(), values.end())
{
}

bool Point::operator==(const Point& other) const
{
  if (values.size() != other.values.size())
    return false;

  for (size_t i = 0, e = rank(); i < e; ++i)
    if (values[i] != other.values[i])
      return false;

  return true;
}

bool Point::operator!=(const Point& other) const
{
  if (values.size() != other.values.size())
    return true;

  for (size_t i = 0, e = rank(); i < e; ++i)
    if (values[i] != other.values[i])
      return true;

  return false;
}

Point::data_type Point::operator[](size_t index) const
{
  assert(index < values.size());
  return values[index];
}

size_t Point::rank() const
{
  return values.size();
}

Point::const_iterator Point::begin() const
{
  return values.begin();
}

Point::const_iterator Point::end() const
{
  return values.end();
}

namespace marco::matching
{
  std::ostream& operator<<(std::ostream& stream, const Point& obj)
  {
    stream << "(";

    for (size_t i = 0, e = obj.rank(); i < e; ++i)
    {
      if (i != 0)
        stream << ",";

      stream << obj[i];
    }

    stream << ")";
    return stream;
  }
}