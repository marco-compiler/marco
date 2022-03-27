#ifndef RYU
#define RYU

#include <Windows.h>
#include "marco/runtime/ryuprintf/ryu.h"

#ifndef MSVC_BUILD
inline size_t strlen(const char* s)
{
	size_t i = 0;
	while (*(s + i) != '\0')
		i++;
	return i;
}
#endif

inline char* strncpy(char* dest, const char* src, size_t n)
{
	size_t i;

	for (i = 0; i < n; i++)
	{
		*(dest + i) = *(src + i);
	}
	*(dest + i) = 0;

	return dest;
}

inline int printString(const char* str)
{
	int len = strlen(str);
	HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
	WriteConsoleA(out, str, len, NULL, NULL);
	return len;
}

inline void swap(char* first, char* second)
{
	char tmp = *second;
	*second = *first;
	*first = tmp;
}

inline void reverse(char str[], int length)
{
	int start = 0;
	int end = length - 1;
	while (start < end)
	{
		swap(str + start, str + end);
		start++;
		end--;
	}
}

inline int i2s_buffered_n(int value, char * ptr)
{
	char* str = ptr;
	int i = 0;
	bool neg = false;
	int base = 10;

	/* Handle 0 explicitly, otherwise empty string is printed for 0 */
	if (value == 0)
	{
		str[0] = '0';
		str[1] = '\0';
		return 1;
	}

	// In standard itoa(), negative numbers are handled only with
	// base 10. Otherwise numbers are considered unsigned.
	if (value < 0)
	{
		neg = true;
		value = -value;
	}

	// Process individual digits
	while (value != 0)
	{
		int rem = value % base;
		str[i++] = rem + '0';
		value = value / base;
	}

	// If number is negative, append '-'
	if (neg)
		str[i++] = '-';

	// Reverse the string
	reverse(str, i);

	return i;
}

inline void i2s_buffered(int value, char* ptr)
{
	const int index = i2s_buffered_n(value, ptr);
	ptr[index] = '\0';
}

inline char* i2s(int value)
{
	char* str = (char*) HeapAlloc(GetProcessHeap(), 0x0, 25);
	i2s_buffered(value, str);

	return str;
}

inline uint32_t s2i(const char* s)
{
	uint32_t tmp = 0;
	int i = 0;
	while(1)
	{
		if(*(s+i) == '\0')
			break;
		tmp = tmp * 10;
		tmp = tmp + (*(s+i) - '0');
		i = i + 1;
	}
	return tmp;
}

inline const char* findPercNull(const char* format)
{
    const char* char_ptr;
    unsigned char c = '%';

    for (char_ptr = (const char*)format; ; ++char_ptr)
        if (*char_ptr == c || *char_ptr == '\0')
            return (const char*)char_ptr;
}

#define BUF_SIZE 100
#define TMP_SIZE 25
inline char* composeString(const char* format, va_list ap)
{
	int size = BUF_SIZE;
	int num_chars = 0;
	const char* fmtptr;
	char* bufptr;
	const char* oldfmtptr;
	char* buf = (char*)HeapAlloc(GetProcessHeap(), 0x0, sizeof(char)*size);
	int n;
	char tmp[TMP_SIZE];
	char * tmp_ptr = tmp;
	char precision_str[TMP_SIZE];
	int precision_str_length;
	uint32_t precision = 12;

	fmtptr = findPercNull(format);

	n = fmtptr - format;
	num_chars = num_chars + n;
	while(num_chars > size) {
		size = size * 2;
		buf = (char*)HeapReAlloc(GetProcessHeap(), 0x0, buf, sizeof(char)*size);
	}
	strncpy(buf, format, n);

	bufptr = buf + n;

	while (*fmtptr == '%')
	{
		fmtptr++;
		precision_str_length = 0;
		if(*fmtptr == '.')
		{
			fmtptr++;
			while(*fmtptr >= '0' && *fmtptr <= '9')
			{
				precision_str[precision_str_length] = *fmtptr;
				precision_str_length++;
				fmtptr++;
			}
			precision_str[precision_str_length] = '\0';
			precision = s2i(precision_str);
		}

		if (*fmtptr == 'd')
		{
			tmp_ptr = tmp;
			i2s_buffered(va_arg(ap, int), tmp_ptr);
		}
		else if (*fmtptr == 'l')
		{
			fmtptr++;
			if(*fmtptr == 'd')
			{
				tmp_ptr = tmp;
				i2s_buffered(va_arg(ap, int), tmp_ptr);
			}

		}
		else if (*fmtptr == 'f')
		{
			tmp_ptr = tmp;
			d2fixed_buffered(va_arg(ap, double), precision, tmp_ptr);
		}
		else if (*fmtptr == 's') 
		{
			tmp_ptr = va_arg(ap, char*);
		}
		else
			printString("Unknown placeholder");
		
		n = strlen(tmp_ptr);
		if (n > 0) {
			num_chars = num_chars + n;
			while(num_chars > size) {
				size = size * 2;
				buf = (char*)HeapReAlloc(GetProcessHeap(), 0x0, buf, sizeof(char)*size);
			}
			strncpy(bufptr, tmp_ptr, n);
			bufptr = bufptr + n;
		}

		fmtptr++;
		oldfmtptr = fmtptr;
		fmtptr = findPercNull(fmtptr);

		n = fmtptr - oldfmtptr;
		num_chars = num_chars + n;
		while(num_chars > size) {
			size = size * 2;
			buf = (char*)HeapReAlloc(GetProcessHeap(), 0x0, buf, sizeof(char)*size);
		}
		strncpy(bufptr, oldfmtptr, n);
		bufptr = bufptr + n;
	}

	*(bufptr) = '\0';

    return buf;
}

inline int ryuPrintfInternal(const char* format, va_list ap) {
	char* buf;
	int len;

	buf = composeString(format, ap);
    printString(buf);
	len = strlen(buf);
	HeapFree(GetProcessHeap(), 0x0, buf);

    return len;
}

inline int ryuPrintf(const char* format, ...)
{
	va_list arg;
	int done;

	va_start(arg, format);
	done = ryuPrintfInternal(format, arg);
	va_end(arg);

	return done;
}

inline void printFloat(float value) {
	char str[25];
	d2fixed_buffered(value, 12, str);
	printString(str);
}

inline void printDouble(double value) {
	char str[25];
	d2fixed_buffered(value, 12, str);
	printString(str);
}

inline void printInt(int value) {
	char str[25];
	i2s_buffered(value, str);
	printString(str);
}

inline void printBool(bool value) {
	if(value)
		printString("True\n");
	else
		printString("False\n");
}

inline void printChar(int c) {
	HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
	WriteConsoleA(out, &c, 1, NULL, NULL);
}

template<typename T>
inline void print(T value)
{
  printString("Error: trying to print an unknown type\n");
}

template<>
inline void print<bool>(bool value)
{
  printBool(value);
}

template<>
inline void print<int32_t>(int32_t value)
{
  printInt(value);
}

template<>
inline void print<int64_t>(int64_t value)
{
  printInt(value);
}

template<>
inline void print<float>(float value)
{
  printFloat(value);
}

template<>
inline void print<double>(double value)
{
  printDouble(value);
}
#endif // RYU