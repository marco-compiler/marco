#ifndef RYU
#define RYU

#include <Windows.h>
#include "ryuprintf/d2s.h"
#include "ryuprintf/f2s.h"

inline size_t strlen(const char* s)
{
	size_t i = 0;
	while (*(s + i) != '\0')
		i++;
	return i;
}

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

inline char* i2s(int value)
{
	char* str = (char*) HeapAlloc(GetProcessHeap(), 0x0, 25);
	int i = 0;
	bool neg = false;
	int base = 10;

	/* Handle 0 explicitly, otherwise empty string is printed for 0 */
	if (value == 0)
	{
		str[0] = '0';
		str[1] = '\0';
		return str;
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

	str[i] = '\0';	// Append string terminator

	// Reverse the string
	reverse(str, i);

	return str;
}

inline const char* findPercNull(const char* format)
{
    const char* char_ptr;
    unsigned char c = '%';

    for (char_ptr = (const char*)format; ; ++char_ptr)
        if (*char_ptr == c || *char_ptr == '\0')
            return (const char*)char_ptr;
}

#define SIZE 1000
inline char* composeString(const char* format, va_list ap)
{
	//TODO: realloc if len of buf > SIZE with SIZE*2 and so on
	const char* fmtptr;
	char* bufptr;
	const char* oldfmtptr;
	char* buf = (char*)HeapAlloc(GetProcessHeap(), 0x0, sizeof(char)*SIZE);
	char dummy = '\0';
	char* tmp = &dummy;
	int n;


	fmtptr = findPercNull(format);

	n = fmtptr - format;
	strncpy(buf, format, n);

	bufptr = buf + n;

	while (*fmtptr == '%')
	{
		fmtptr++;
		if (*fmtptr == 'd')
		{
			tmp = i2s(va_arg(ap, int));
		}
		else if (*fmtptr == 'f')
		{
			tmp = d2s(va_arg(ap, double));
		}
		else if (*fmtptr == 's') 
		{
			tmp = va_arg(ap, char*);
		}
		
		n = strlen(tmp);
		if (n > 0) {
			strncpy(bufptr, tmp, n);
			if(*fmtptr != 's')
				HeapFree(GetProcessHeap(), 0x0, tmp);
			bufptr = bufptr + n;
		}

		fmtptr++;
		oldfmtptr = fmtptr;
		fmtptr = findPercNull(fmtptr);

		n = fmtptr - oldfmtptr;
		strncpy(bufptr, oldfmtptr, n);
		bufptr = bufptr + n;
	}

	*bufptr = '\0';

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

inline int runtimePrintf(const char* format, ...)
{
	va_list arg;
	int done;

	va_start(arg, format);
	done = ryuPrintfInternal(format, arg);
	va_end(arg);

	return done;
}

inline void printFloat(float value) {
	char* str = f2s(value);
	printString(str);
	HeapFree(GetProcessHeap(), 0x0, str);
}

inline void printDouble(double value) {
	char* str = d2s(value);
	printString(str);
	HeapFree(GetProcessHeap(), 0x0, str);
}

inline void printInt(int value) {
	char* str = i2s(value);
	printString(str);
	HeapFree(GetProcessHeap(), 0x0, str);
}

inline void printBool(bool value) {
	if(value)
		printString("True\n");
	else
		printString("False\n");
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