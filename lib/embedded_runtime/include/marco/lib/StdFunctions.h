#ifndef STD_FUNCTIONS_H
#define STD_FUNCTIONS_H

#include "../driver/heap.h"


namespace stde{
    //CMATH
    int pow(int base, int exponent);
    float pow(float base, float exponent);
    float abs(float value);
    unsigned int abs(int value);
    float sqrt(float value);
    float acos(float x);
    float asin(float x);
    float log(float x);
    float log10(float x);
    float exp(float x);
    float cosh(float x);
    float atan(float value);
    float atan2(float x, float y);
    int max(int x , int y);
    float max(float x, float y);
    float min(float x, float y);
    int min(int x, int y);
    float sinh(float value);
    

    //TODO
    
   
    float cos(float value);
    float sin(float value);
    
    float tan(float value);
    float tanh(float value);

    //ALGORITHM
    template <class InputIterator, class UnaryPredicate>
    bool all_of( InputIterator first, InputIterator last, UnaryPredicate pred)
    {
          while (first!=last) {
            if (!pred(*first)) return false;
        ++first;
        }
    return true;
    }

    template<class ForwardIt>
    ForwardIt max_element(ForwardIt first, ForwardIt last)
    {
    if (first == last) return last;
 
    ForwardIt largest = first;
    ++first;
    for (; first != last; ++first) {
        if (*largest < *first) {
            largest = first;
        }
    }
    return largest;
    }

    template< class T >
    struct plus{
        constexpr T operator()(const T &lhs, const T &rhs) const 
    {
        return lhs + rhs;
    }
    };

    template< class T >
    struct multiplies{
        constexpr T operator()(const T &lhs, const T &rhs) const 
    {
        return lhs * rhs;
    }
    };


    
    template<class InputIterator, class UnaryPredicate>
  bool any_of (InputIterator first, InputIterator last, UnaryPredicate pred)
{
  while (first!=last) {
    if (pred(*first)) return true;
    ++first;
  }
  return false;
}

template<class ForwardIt>
ForwardIt min_element(ForwardIt first, ForwardIt last)
{
    if (first == last) return last;
 
    ForwardIt smallest = first;
    ++first;
    for (; first != last; ++first) {
        if (*first < *smallest) {
            smallest = first;
        }
    }
    return smallest;
}

template<class InputIt, class T>
constexpr // since C++20
T accumulate(InputIt first, InputIt last, T init)
{
    for (; first != last; ++first) {
        T temp = init;
        init = temp + *first; // std::move since C++20
    }
    return init;
}

    //CASSERT

    void assert(bool cond);
    //VECTOR
    template<class T>
    class Vector
    {
    private:
    /* data */
       
    public:
    typedef struct node
    {
        /* data */
        T data;
        struct node* next;

    }Node;
    
    unsigned int size;
    Node* head;
    Node* last;
    Node* second_last;

    Vector(){
        this->head = nullptr;
        this->last = head;
        this->second_last = head;
    }

    void push_back(T data){
        Node* alloc = (Node*) malloc(sizeof(Node));
        alloc->data = data;
        if(head == nullptr){ //First 
            head = alloc;
            last = head;
            return;
        }
        last->next = alloc;
        if(head == last){ //Second
            last = last->next;
            second_last = head;
            return;
        }
            second_last = last;
            last = last->next;
        
        return;
    }

    };

    template<class T, unsigned int N>
    class array{
        private:
        const static unsigned int capacity = N;
        unsigned int num_elements ;
        T arr[capacity];

        public:
        array() : num_elements{0} {

        }

        T* data(){
            return arr;
        }

        unsigned int size(){
            return num_elements;
        }
    };

    template<bool B, class T = void>
    struct enable_if {};

    template<class T>
    struct enable_if<true, T> { typedef T type; };


    struct ostream{}; //stub structure in order to allow printing functions work.
 
}
    
#endif
