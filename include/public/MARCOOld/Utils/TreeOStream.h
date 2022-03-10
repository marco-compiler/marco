#ifndef MARCO_UTILS_TREEOSTREAM_H
#define MARCO_UTILS_TREEOSTREAM_H

#include <iostream>

namespace marco::utils
{
  class TreeOStream;

  class TreeOStream : public std::ostream
  {
    private:
    class Buffer : public std::streambuf
    {
      public:
      Buffer(std::streambuf* dest)
              : myDest(dest),
                lineIndicator(" |"),
                propertyIndicator("--"),
                newLine(false),
                property(false)
      {
      }

      void setProperty()
      {
        property = true;
      }

      protected:
      int overflow(int ch) override
      {
        int result = 0;

        if (ch != traits_type::eof())
        {
          if (newLine)
          {
            newLine = false;
            myDest->sputn(lineIndicator.data(), lineIndicator.size());

            if (property)
            {
              myDest->sputn(propertyIndicator.data(), propertyIndicator.size());
            }
            else
            {
              for (size_t i = 0, e = propertyIndicator.size(); i < e; ++i)
                myDest->sputc(' ');
            }

            myDest->sputc(' ');
          }

          result = myDest->sputc(ch);

          if (ch == '\n') {
            newLine = true;
            property = false;
          }
        }

        return result;
      }

      private:
      std::streambuf* myDest;
      std::string lineIndicator;
      std::string propertyIndicator;
      bool newLine;
      bool property;
    };

    private:
    using Base = std::ostream;
    using buffer_type = Buffer;

    public:
    using stream_type = std::ostream;
    using char_type = typename Base::char_type;
    using traits_type = typename Base::traits_type;
    using int_type = typename Base::int_type;
    using pos_type = typename Base::pos_type;
    using off_type = typename Base::off_type;

    public:
    TreeOStream(stream_type& stream)
          : Base(&buffer), buffer(buffer_type(stream.rdbuf()))
    {
    }

    void setPropertyFlag()
    {
      buffer.setProperty();
    }

    private:
    TreeOStream(const TreeOStream&) = delete;
    const TreeOStream& operator=(const TreeOStream&) = delete;

    private:
    buffer_type buffer;
  };

  class tree_property_t {};
  constexpr tree_property_t tree_property;

  inline TreeOStream& operator<<(TreeOStream& os, tree_property_t)
  {
    os.setPropertyFlag();
    return os;
  }
}

#endif //MARCO_UTILS_TREEOSTREAM_H
